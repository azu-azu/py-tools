"""2つのCSV出力を突合するスクリプト。

基準となる正解データをgolden、検証対象の出力をtargetと呼ぶ。

移行前後の出力比較や、リファクタ前後の回帰確認などに使う。
"""

from __future__ import annotations

import argparse
import bisect
import re
import traceback
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────
# デフォルト設定

DATA_DIR = Path(__file__).resolve().parents[1]

# 以下はサンプル値
# 実際に使うgolden/target/キー列に合わせて手動で書き換える

# golden CSVのstem
# 引数を省略したときに使われる
DEFAULT_GOLDEN_NAME = "golden"

# golden CSVを探すディレクトリ
DEFAULT_GOLDEN_DIR = DATA_DIR / "sample"

# target CSVのデフォルトパス
DEFAULT_TARGET = DATA_DIR / "output" / "debug" / "_test.csv"

# 突合キー列
# 空リストにするとキーなしモードで比較する
DEFAULT_KEYS: list[str] = ["ID"]


# ────────────────────────────────────────────────────────────────────
# ファイル固有の設定

# 文字化けしているカラム
# 文字化けによるdiff検出を避けたい場合は、該当カラムを指定する
GARBLED_COLS: list[str] = []

# 日付型カラム
# mm/dd と m/d の表示違いによるdiff検出を避けたい場合に指定する
# 自動検出で拾えないカラムだけを手動指定する
DATE_COLS: list[str] = []

# only_right表示時に追加するカラム
EXTRA_COLS: list[str] = []


# ────────────────────────────────────────────────────────────────────
# 比較設定

LEFT_KEY = "golden"
RIGHT_KEY = "target"

# 差分表示の既定行数
# 実行ごとに変えたい場合はrun_verifyのmax_rows引数で上書きする
DEFAULT_MAX_ROWS: int = 20

FLOAT_ATOL: float = 1e-9

ALNUM_PATTERN = re.compile(r"[0-9A-Za-z]+")

DEFAULT_ENCODINGS: tuple[str, ...] = (
    "cp932",
    "shift_jis_2004",
    "utf-8-sig",
)


# ────────────────────────────────────────────────────────────────────
# CSV読み込み

def _read_csv(path: Path, label: str) -> pd.DataFrame:
    """encodingを順に試してCSVを読む。

    全滅した場合は最後のencodingで強制オープンする。
    """
    for encoding in DEFAULT_ENCODINGS:
        try:
            # low_memory=Trueだとチャンクごとに型を推論するため、
            # 同じ列にintとstrが混ざったobject列ができることがある。
            # どこで型が割れるかは行数まかせで左右非対称になるので、
            # 列全体で1回だけ推論させる。
            #
            # 列全体を保持する分ピークメモリは増えるが、
            # 突合では型のブレを消すほうを優先する。
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
        except (UnicodeDecodeError, LookupError):
            print(f"⚠️ {label}: encoding={encoding} NG")
        else:
            print(f"✅ {label}: encoding={encoding}")
            return df

    last = DEFAULT_ENCODINGS[-1]

    print(
        f"⚠️ {label}: 全encoding失敗、"
        f"{last} で強制オープン（文字化けの可能性あり）"
    )

    return pd.read_csv(
        path,
        encoding=last,
        encoding_errors="replace",
        low_memory=False,
    )


# ────────────────────────────────────────────────────────────────────
# 文字列カラムの選択

def _text_cols(df: pd.DataFrame) -> pd.Index:
    """文字列として扱うカラム名を返す。

    文字列カラムのdtypeは、pandas 2ではobject、pandas 3ではstrになる。

    pandas 3のinclude="object"は後方互換でstrも拾うが非推奨警告が出る。
    一方、pandas 2のinclude="str"はTypeErrorになる。

    そのため["object", "str"]を先に試し、
    弾かれたpandas 2ではobjectのみにフォールバックする。
    """
    try:
        return df.select_dtypes(include=["object", "str"]).columns
    except TypeError:
        return df.select_dtypes(include="object").columns


# ────────────────────────────────────────────────────────────────────
# 日付列の自動判定

def _detect_date_cols(
    df: pd.DataFrame,
    min_ratio: float = 0.95,
    sample: int = 2000,
) -> list[str]:
    """文字列カラムのうち、値の大半が日付として解釈できる列名を返す。

    左右で個別に判定すると非対称になり、偽差分の原因になるため、
    golden側だけに対して呼び出す。

    純粋な数値列は、IDやコードを日付と誤判定しないよう除外する。
    """
    cols: list[str] = []

    for col in _text_cols(df):
        values = df[col].dropna().astype(str).str.strip()
        values = values[values != ""]

        if values.empty:
            continue

        # IDや郵便番号などの純粋な数値列は日付扱いしない
        if values.str.fullmatch(r"\d+").all():
            continue

        sampled = values.sample(
            min(len(values), sample),
            random_state=0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            ratio = (
                pd.to_datetime(
                    sampled,
                    errors="coerce",
                )
                .notna()
                .mean()
            )

        if ratio >= min_ratio:
            cols.append(col)

    return cols


def _resolve_date_cols(reference: pd.DataFrame) -> list[str]:
    """手動指定と自動判定を統合した日付列リストを返す。"""
    manual = [col for col in DATE_COLS if col]
    auto = _detect_date_cols(reference)

    # 順序を維持したまま重複を除去
    return list(dict.fromkeys(manual + auto))


# ────────────────────────────────────────────────────────────────────
# パス解決

def _resolve_golden_path(golden_name: str | None) -> Path:
    """goldenのstemから、固定フォルダ内のCSVパスを作る。

    Noneまたは空文字の場合は、DEFAULT_GOLDEN_NAMEを使用する。
    """
    resolved_name = golden_name or DEFAULT_GOLDEN_NAME
    stem = Path(resolved_name).stem

    if not stem:
        raise ValueError(
            "golden CSVの名前が空です。"
            "引数で指定するか、DEFAULT_GOLDEN_NAMEを設定してください"
        )

    return DEFAULT_GOLDEN_DIR / f"{stem}.csv"


# ────────────────────────────────────────────────────────────────────
# 正規化

def _sort_by_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """列名の辞書順を優先キーにして行を並べ替える。

    突合結果を実行ごとに安定させるための決定的な並び替え。

    元ファイルの行順はここで失われるため、
    並び順を比較したい場合はこれを通す前の状態を使う。
    """
    if len(df.columns) == 0:
        return df.reset_index(drop=True)

    return (
        df
        .sort_values(sorted(df.columns))
        .reset_index(drop=True)
    )


def _normalize(
    df: pd.DataFrame,
    date_cols: list[str],
    *,
    sort_rows: bool = True,
) -> pd.DataFrame:
    """突合前のDataFrameを正規化する。

    sort_rowsをFalseにすると、元ファイルの行順のまま返す。
    """
    normalized = df.copy()

    # 文字列カラムの前後空白を除去
    #
    # object列は「中身が全部str」とは限らない。
    # read_csvはチャンク単位で型を推論するため、同じ列の中に
    # int と str が混ざったobject列ができることがある。
    #
    # 混在した列に.str.strip()を列ごと掛けると、str以外のセルは
    # 値を保たずNaNになる。直後のfillna("")がそれを空文字にするので、
    # 元の値が丸ごと消えて偽差分になる。
    # さらに全セルがintのobject列では.strアクセサ自体が
    # AttributeErrorで落ちる。
    #
    # そのためstrのセルだけを個別にstripする。
    # 純粋な文字列列は従来どおりベクトル化された.str.strip()で処理する。
    for col in _text_cols(normalized):
        series = normalized[col]

        inferred = pd.api.types.infer_dtype(series, skipna=True)

        if inferred in ("string", "empty"):
            normalized[col] = series.str.strip()
            continue

        normalized[col] = series.map(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else value
            )
        )

    # NaNと空文字を統一
    # 数値列まで空文字で埋めるとobject型になり、
    # np.iscloseによる数値比較が効かなくなるため対象外とする
    text_cols = _text_cols(normalized)
    normalized[text_cols] = normalized[text_cols].fillna("")

    # 日付カラムを統一フォーマットに正規化
    for col in date_cols:
        if col not in normalized.columns:
            continue

        parsed = pd.to_datetime(
            normalized[col],
            errors="coerce",
        )

        lost = (
            normalized[col].notna()
            & normalized[col].ne("")
            & parsed.isna()
        )

        if lost.any():
            print(
                f"⚠️ {col}: "
                f"日付として解釈できない値 {lost.sum()}件"
            )
            print(
                normalized.loc[lost, col]
                .drop_duplicates()
                .head(10)
                .to_string()
            )

        normalized[col] = parsed.dt.strftime("%Y-%m-%d")

    # 日付正規化によって新たに発生したNaNを吸収
    text_cols = _text_cols(normalized)
    normalized[text_cols] = normalized[text_cols].fillna("")

    normalized = normalized.reset_index(drop=True)

    return (
        _sort_by_column_names(normalized)
        if sort_rows
        else normalized
    )


def _ascii_signature(value: str) -> str:
    """文字列からASCII英数字だけを抽出したsignatureを返す。"""
    normalized = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    return "".join(ALNUM_PATTERN.findall(normalized))


# ────────────────────────────────────────────────────────────────────
# dtypeの整合

def _is_numeric_col(series: pd.Series) -> bool:
    """mergeキーとして数値扱いしてよい列かを返す。

    boolはis_numeric_dtypeがTrueになるが、
    文字列化するとTrue/Falseになり数値表記と揃わないため除外する。
    """
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def _is_text_col(series: pd.Series) -> bool:
    """文字列として扱うdtypeかを返す。

    文字列カラムのdtypeはpandas 2ではobject、pandas 3ではstrになるが、
    この2つは混在したままでもmergeできるので同一視する。
    """
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _to_text(series: pd.Series) -> pd.Series:
    """左右で表記が揃うように文字列化する。

    astype(str)任せにすると2種類の偽差分が出る。

    1つ目は欠損で、pandas 2では"nan"、pandas 3では欠損のまま残り、
    _normalizeで空文字にした文字列側と食い違う。

    2つ目は整数値で、欠損が1つあるだけで列がfloat64になり、
    1 が "1.0" になって文字列側の "1" と食い違う。

    そのため欠損は空文字へ、整数値の浮動小数点数は整数表記へ寄せる。
    """
    if _is_numeric_col(series):
        numbers = pd.to_numeric(series, errors="coerce")

        # 1e15を超えると整数へ丸めた時点で桁が落ちるため対象外にする
        integral = (
            numbers.notna()
            & (numbers % 1 == 0)
            & (numbers.abs() < 1e15)
        )
        fractional = numbers.notna() & ~integral

        text = pd.Series("", index=series.index, dtype=object)
        text[integral] = (
            numbers[integral].astype("int64").astype(str)
        )
        text[fractional] = numbers[fractional].astype(str)

        return text.astype(str)

    if pd.api.types.is_datetime64_any_dtype(series):
        # 時刻を持たない日付型は、_normalizeの日付列と同じ表記へ寄せる
        if (series.dropna().dt.normalize() == series.dropna()).all():
            formatted = series.dt.strftime("%Y-%m-%d")
        else:
            formatted = series.astype(str)

        return formatted.where(series.notna(), "").astype(str)

    text = series.astype(object)

    return text.where(series.notna(), "").astype(str)


def _align_common_dtypes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """左右でdtypeが食い違う共通列を、mergeできる型へ揃える。

    read_csvはファイル単位で型を推論するため、同じ列でも
    golden側がint64、target側がstrになることがある。

    共通列はすべてStage 1のmergeキーになるので、
    1列でも型が割れているとmerge前にValueErrorで落ちる。

    dtypeが同じ列、数値どうし(int64 vs float64)、
    文字列どうし(object vs str)は、そのままmergeできるので触らない。

    boolと数値の組み合わせは、文字列化するとTrue/Falseと1/0になって
    全行が食い違うため、bool側を数値へ寄せる。

    片方だけ数値の場合、文字列側が全部数値として読めるなら数値へ寄せ、
    読めない値があるなら両方を文字列へ落とす。

    数値へ寄せた列はFLOAT_ATOLによる近似比較が残るが、
    文字列化した列は表記そのものの比較になるため、
    どちらへ倒れたかを毎回printする。
    """
    # 列ごと差し替えるだけで元の値は書き換えないため、浅いコピーで足りる
    left = left.copy(deep=False)
    right = right.copy(deep=False)

    for col in cols:
        left_col = left[col]
        right_col = right[col]

        # 同じdtypeなら揃える余地がない
        if left_col.dtype == right_col.dtype:
            continue

        left_numeric = _is_numeric_col(left_col)
        right_numeric = _is_numeric_col(right_col)

        # int64 vs float64 はmergeも比較も問題なく通る
        if left_numeric and right_numeric:
            continue

        # object vs str はそのままmergeできる
        if (
            _is_text_col(left_col)
            and _is_text_col(right_col)
        ):
            continue

        left_bool = pd.api.types.is_bool_dtype(left_col)
        right_bool = pd.api.types.is_bool_dtype(right_col)

        # bool vs 数値
        # 欠損を持つboolも通せるようnullableなInt64へ寄せる
        if (
            (left_bool and right_numeric)
            or (right_bool and left_numeric)
        ):
            bool_frame = left if left_bool else right
            bool_frame[col] = bool_frame[col].astype("Int64")

            print(f"ℹ️ {col}: bool側を数値へ揃えた")
            continue

        numeric_vs_text = (
            left_numeric and _is_text_col(right_col)
        ) or (
            right_numeric and _is_text_col(left_col)
        )

        if not numeric_vs_text:
            # ここまでで拾えなかった組み合わせ
            # 日付型 vs 文字列、bool vs 文字列など
            left[col] = _to_text(left_col)
            right[col] = _to_text(right_col)

            print(f"⚠️ {col}: dtype不一致のため両方を文字列化")
            continue

        text_frame = right if left_numeric else left
        text_side = text_frame[col]

        parsed = pd.to_numeric(text_side, errors="coerce")

        # 空欄はNaNとして数値側の欠損と対応するため、失敗扱いにしない
        blank = (
            text_side.isna()
            | text_side.astype(str).str.strip().eq("")
        )

        if (parsed.notna() | blank).all():
            text_frame[col] = parsed

            print(f"ℹ️ {col}: 文字列側を数値へ揃えた")
            continue

        sample = (
            text_side[parsed.isna() & ~blank]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        left[col] = _to_text(left[col])
        right[col] = _to_text(right[col])

        print(
            f"⚠️ {col}: 型不一致のため両方を文字列化 例: {sample}"
        )

    return left, right


# ────────────────────────────────────────────────────────────────────
# 文字化け行の吸収（キーなしモード用）

def _absorb_garbled_rows(
    resid_left: pd.DataFrame,
    resid_right: pd.DataFrame,
    common_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """文字化けだけが違う行を左右でペアにして差分から取り除く。

    キーがあるモードでは、行をキーで対応づけてからセル単位で
    signature比較を行うが、キーなしモードでは対応づけができない。

    そこで、文字化け対象列をASCII英数字のsignatureへ置き換えたうえで
    行単位の突合をやり直し、ペアになった行を差分から除外する。

    戻り値は (only_left, only_right, fuzzy_matched)。
    """
    garbled_cols = [
        col
        for col in common_cols
        if col in GARBLED_COLS
    ]

    if (
        not garbled_cols
        or resid_left.empty
        or resid_right.empty
    ):
        return (
            resid_left,
            resid_right,
            pd.DataFrame(),
        )

    left_sig = resid_left.copy()
    right_sig = resid_right.copy()

    for col in garbled_cols:
        left_sig[col] = left_sig[col].map(_ascii_signature)
        right_sig[col] = right_sig[col].map(_ascii_signature)

    # signature置換で新たに重複した行も取り違えないよう出現順連番を振る
    left_sig["_gseq"] = (
        left_sig
        .groupby(common_cols, dropna=False, observed=True)
        .cumcount()
    )
    right_sig["_gseq"] = (
        right_sig
        .groupby(common_cols, dropna=False, observed=True)
        .cumcount()
    )

    # 突合後に元の値へ戻れるよう行番号を持たせる
    left_sig["_lrow"] = np.arange(len(left_sig))
    right_sig["_rrow"] = np.arange(len(right_sig))

    matched = left_sig.merge(
        right_sig,
        how="inner",
        on=common_cols + ["_gseq"],
    )

    left_rows = matched["_lrow"].to_numpy()
    right_rows = matched["_rrow"].to_numpy()

    # ペアになった行のうち、実際に値が違うセルだけを記録する
    fuzzy: list[pd.DataFrame] = []

    for col in garbled_cols:
        left_values = pd.Series(
            resid_left[col].to_numpy()[left_rows]
        )
        right_values = pd.Series(
            resid_right[col].to_numpy()[right_rows]
        )

        rescued_mask = (
            left_values.astype(str)
            != right_values.astype(str)
        ).to_numpy()

        if rescued_mask.any():
            fuzzy_frame = pd.DataFrame(
                {
                    "column": col,
                    LEFT_KEY: left_values[rescued_mask],
                    RIGHT_KEY: right_values[rescued_mask],
                }
            )
            fuzzy.append(fuzzy_frame.reset_index(drop=True))

    fuzzy_matched = (
        pd.concat(fuzzy, ignore_index=True)
        if fuzzy
        else pd.DataFrame()
    )

    only_left = (
        resid_left
        .drop(index=resid_left.index[left_rows])
        .reset_index(drop=True)
    )

    only_right = (
        resid_right
        .drop(index=resid_right.index[right_rows])
        .reset_index(drop=True)
    )

    return only_left, only_right, fuzzy_matched


# ────────────────────────────────────────────────────────────────────
# 並び順の比較

# 差分として保持するサンプルの上限
# 表示自体はmax_rowsで絞るが、全件保持しても使い道がないので蓋をする
ORDER_SAMPLE_CAP: int = 1000

# サンプル表示時の1行あたりの文字数上限
ORDER_LABEL_WIDTH: int = 60

# 並び順を判定するのに必要な、対応づいた行数
#
# 1行以下では並びようがないので、順序という概念自体が成立しない。
# これは有用性の閾値ではなく、判定が定義できる下限。
#
# 「対応がN行しかないのに一致と言われても困る」は表示側の問題なので、
# 一致・不一致どちらの行にも対応行数と除外行数を必ず並べて解いている。
ORDER_MIN_PAIRED_ROWS: int = 2

# 「キーが1件も対応していない」と言い切るのに必要な行数
#
# 数行のファイルでは、単に別データなだけでも
# 「1件も対応していない」が数式の上では成立してしまう。
# 突合そのものが壊れている状態だけを拾いたいので下限を置く。
ORDER_ALL_UNMATCHED_MIN_ROWS: int = 10


@dataclass(frozen=True)
class OrderResult:
    """並び順の比較結果。

    行の並び順は、対応づいた行だけを取り出して相対順序で判定する。
    片側にしかない行は比較から除外し、その行数をleft_excluded /
    right_excludedに持つ。

    対応づいた行が足りずに判定できなかった場合はcheckedがFalseになり、
    skip_reasonに理由が入る。

    列の並び順は前提条件なしで判定できるため、checkedとは無関係に埋まる。
    """

    checked: bool
    skip_reason: str
    key_cols: list[str]
    ambiguous_rows: int
    paired_rows: int
    left_excluded: int
    right_excluded: int
    first_diff: int | None
    diff_count: int
    moved_rows: int
    samples: pd.DataFrame
    left_cols: list[str]
    right_cols: list[str]

    @property
    def has_ambiguity(self) -> bool:
        """判定できずに除外した対応がある場合はTrueを返す。

        ambiguous_rowsは除外行数の内訳であり、
        left_excluded / right_excludedに含まれている。
        """
        return self.ambiguous_rows > 0

    @property
    def has_excluded(self) -> bool:
        """比較から除外した行がある場合はTrueを返す。"""
        return self.left_excluded > 0 or self.right_excluded > 0

    @property
    def row_match(self) -> bool:
        """対応づいた行の相対順序が保たれていた場合にTrueを返す。

        対応づかなかった行と、対応が一意に決まらなかった行は
        比較から除外しているため、has_excludedがTrueなら
        「除外した上での一致」になる。

        除外した行がどれだけあるかはleft_excluded /
        right_excludedに入る。行の過不足そのものは
        VerifyResult.is_matchが見る。
        """
        return self.checked and self.diff_count == 0

    @property
    def col_match(self) -> bool:
        """共通列の並び順が一致している場合はTrueを返す。"""
        return self.left_cols == self.right_cols

    @property
    def is_match(self) -> bool:
        """行と列の並び順が両方とも一致と判定された場合にTrueを返す。

        判定できない対応は仮定で埋めずに除外しているため、
        ここに「入れ替わっていないはず」という推測は入らない。
        除外した行数はleft_excluded / right_excludedで見る。
        """
        return self.row_match and self.col_match


def _order_text(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """並び順の比較に使う列を、左右で表記が揃う文字列へ変換する。

    文字化け対象列はASCII英数字のsignatureへ置き換える。

    _absorb_garbled_rowsは文字化けした行をsignatureでペアにして
    差分から取り除くが、生の値は左右で違ったまま残る。

    そのまま比較すると、吸収したはずの行で
    「並び順が違う」と誤検知するため、ここで同じ土俵に乗せる。
    """
    text = {
        col: (
            _to_text(df[col]).map(_ascii_signature)
            if col in GARBLED_COLS
            else _to_text(df[col])
        )
        for col in cols
    }

    return pd.DataFrame(text, index=df.index)


def _order_labels(
    text: pd.DataFrame,
    cols: list[str],
    positions: np.ndarray,
) -> list[str]:
    """サンプル表示用に、行を識別する列の値を1行1文字列へまとめる。

    キーなしモードでは共通列すべてが識別材料になり、
    そのまま並べると1行が横に伸びすぎるため頭で切る。
    """
    labels: list[str] = []
    values = text[cols].to_numpy()

    for row in values[positions]:
        label = " | ".join(row)

        if len(label) > ORDER_LABEL_WIDTH:
            label = label[: ORDER_LABEL_WIDTH - 1] + "…"

        labels.append(label)

    return labels


def _longest_increasing(values: np.ndarray) -> np.ndarray:
    """狭義単調増加となる最長の部分列の添字を返す。

    ここから外れた行が「動かせば順序が揃う行」、
    つまり実際に動いた行になる。

    値がすべて異なることを前提にしている。
    対応づけは単射なので、この配列に重複は入らない。

    最長増加部分列は一意ではないが、長さは一意なので、
    どれを選んでも「動いた行数」は変わらない。
    """
    total = values.size

    if total == 0:
        return np.empty(0, dtype=np.int64)

    # tails[k] は長さk+1の増加列の末尾の添字
    tails: list[int] = []
    tail_values: list[int] = []
    parent = np.full(total, -1, dtype=np.int64)

    for index in range(total):
        value = int(values[index])
        position = bisect.bisect_left(tail_values, value)

        if position:
            parent[index] = tails[position - 1]

        if position == len(tails):
            tails.append(index)
            tail_values.append(value)
        else:
            tails[position] = index
            tail_values[position] = value

    chain: list[int] = []
    node = tails[-1]

    while node >= 0:
        chain.append(node)
        node = int(parent[node])

    return np.array(chain[::-1], dtype=np.int64)


def _pair_positions(
    left_text: pd.DataFrame,
    right_text: pd.DataFrame,
    key_cols: list[str],
) -> tuple[np.ndarray, int]:
    """左の各行に対応する右の行位置と、対応が一意に決まらない行数を返す。

    対応づかなかった行は-1のまま残る。

    Stage 1 / Stage 2 と同じ2段構えを、元の行位置を保ったまま再現する。

    キー列の並びを直接比べる方式では、同じキーが複数行あるときに
    キー内での入れ替わりが見えない。左右どちらも同じキーが並ぶため、
    キー列だけを見ると入れ替わっていても一致に見えてしまう。

    そこで「どの行がどの行とペアになったか」を作り、
    その対応の相対順序で並び順を判定する。

    Pass Aは全列一致でペアにする。内容で結ぶため、
    同じキーが複数行あっても、どれとどれが対応するかを取り違えない。

    Pass Bは残差をキーで対応づける。値が違っても
    キーさえ対応していれば順序は判定できる。

    どちらもinner mergeなので、左右の行数が違っても、
    片側にしかない行があっても、対応づいた分だけが残る。
    ペアリングは全単射である必要がなく、単射であれば足りる。

    キーなしモードではPass Bを走らせない。
    行を対応づける手段がないため、Pass Aで結べなかった行は
    そのまま除外される。値が1セル違うだけの行もここに落ちる。

    Pass Aの groupby は、Stage 1 が済ませた仕事を
    文字列化した値でもう一度やる形になっている。

    速度が問題になったときの犯人はここだが、潰すには Stage 1 へ
    行位置を持たせてその結果を再利用することになり、
    突合の本体に並び順の都合が染み出す。

    並び順の比較をこのブロックだけで完結させておくほうが、
    壊れたときの切り分けが効くため、重複のまま残している。
    """
    cols = list(left_text.columns)
    left_total = len(left_text)
    right_total = len(right_text)

    left = left_text.reset_index(drop=True)
    right = right_text.reset_index(drop=True)

    left["_lrow"] = np.arange(left_total)
    right["_rrow"] = np.arange(right_total)

    # ────────────────────────────────────────────────────────────────
    # Pass A: 全列一致

    left["_cseq"] = (
        left
        .groupby(cols, dropna=False, observed=True)
        .cumcount()
    )
    right["_cseq"] = (
        right
        .groupby(cols, dropna=False, observed=True)
        .cumcount()
    )

    # 同じ内容の行が左右それぞれ何行あるか
    # mergeで両方ともペアの行についてくる
    left["_lcount"] = (
        left
        .groupby(cols, dropna=False, observed=True)
        ["_cseq"]
        .transform("size")
    )
    right["_rcount"] = (
        right
        .groupby(cols, dropna=False, observed=True)
        ["_cseq"]
        .transform("size")
    )

    exact = left.merge(
        right,
        how="inner",
        on=cols + ["_cseq"],
    )

    # 内容グループの行数が左右で一致するペアだけを採用する
    #
    # 一致していれば、出現順のペアは位置順のペアと同じになる。
    # 内容が同じ行どうしの入れ替えは観測できないので、
    # グループ内でどの対応を選んでも等価であり、交差も生まない。
    #
    # 行数が違う場合は「どの行をあぶれさせるか」で対応先が変わる。
    #
    #   golden: (1,1)@0, (1,1)@6
    #   target: (1,101)@0, (1,1)@6
    #
    # 出現順に取ると golden@0 が target@6 と結ばれて交差するが、
    # golden@6 を採れば順序は保たれる。どちらが正しいかは
    # 情報がなく、正しく解くにはキー群ごとの系列アライメントが要る。
    # キーなしモードでは全ファイルが1グループになりうるため現実的でない。
    #
    # 判定できないものを一致にも不一致にも混ぜず、除外側へ回す。
    forced = (
        exact["_lcount"] == exact["_rcount"]
    ).to_numpy()

    mapped = np.full(left_total, -1, dtype=np.int64)
    mapped[exact.loc[forced, "_lrow"].to_numpy()] = (
        exact.loc[forced, "_rrow"].to_numpy()
    )

    # Pass Aで採用しなかったペアの数
    #
    # キーなしモードではPass Bを通らないため、これがそのまま曖昧な行になる。
    #
    # キーありモードでは、ここで落ちた行は必ずPass Bの残差に入り、
    # 同じキーの残差が2行以上ある側でもう一度落ちる。
    # 二重に数えないよう、Pass Bの数で置き換える。
    ambiguous_rows = int((~forced).sum())

    if not key_cols:
        return mapped, ambiguous_rows

    # ────────────────────────────────────────────────────────────────
    # Pass B: 残差をキーで対応づけ

    paired_right = np.zeros(right_total, dtype=bool)
    paired_right[mapped[mapped >= 0]] = True

    rest_left = left.loc[mapped < 0, key_cols + ["_lrow"]].copy()
    rest_right = right.loc[~paired_right, key_cols + ["_rrow"]].copy()

    # 片側の残差が空なら対応づけようがない
    #
    # Pass Aで落ちた行があれば相手側の行も必ず残るため、
    # ここへ来る時点でambiguous_rowsは0になっている。
    if rest_left.empty or rest_right.empty:
        return mapped, ambiguous_rows

    rest_left["_kseq"] = (
        rest_left
        .groupby(key_cols, dropna=False, observed=True)
        .cumcount()
    )
    rest_right["_kseq"] = (
        rest_right
        .groupby(key_cols, dropna=False, observed=True)
        .cumcount()
    )

    # 同じキーの残差が何行あるかを左右それぞれで持たせる
    # mergeで両方ともペアの行についてくる
    rest_left["_lsize"] = (
        rest_left
        .groupby(key_cols, dropna=False, observed=True)
        ["_kseq"]
        .transform("size")
    )
    rest_right["_rsize"] = (
        rest_right
        .groupby(key_cols, dropna=False, observed=True)
        ["_kseq"]
        .transform("size")
    )

    keyed_pairs = rest_left.merge(
        rest_right,
        how="inner",
        on=key_cols + ["_kseq"],
    )

    # 残差の同じキーが左右どちらかで2行以上あるペアは、対応が決まらない
    #
    # Pass Bは出現順にペアにするしかない。
    # 同一キーで両側に値差分があると情報がなく、
    # 入れ替わったのか値が変わったのか原理的に区別できない。
    #
    # 左右どちらも1行なら対応は強制されるので曖昧さはない。
    # 左右で残差の行数が違いうるので、片側だけを見ると取りこぼす。
    #
    # 曖昧なペアは採用せず、除外側へ回す。
    #
    # 出現順で結んで「入れ替わっていない」と仮定すると、
    # Pass Aがどの行を取ったかによって、順序が保たれている場合でも
    # 交差として現れることがある。判定できないものを
    # 一致にも不一致にも混ぜないほうが、他の扱いと揃う。
    resolved = (
        keyed_pairs[["_lsize", "_rsize"]].max(axis=1) <= 1
    ).to_numpy()

    mapped[keyed_pairs.loc[resolved, "_lrow"].to_numpy()] = (
        keyed_pairs.loc[resolved, "_rrow"].to_numpy()
    )

    return mapped, int((~resolved).sum())


def _skip_order(
    reason: str,
    key_cols: list[str],
    left_cols: list[str],
    right_cols: list[str],
    ambiguous_rows: int = 0,
) -> OrderResult:
    """行の並び順を判定しなかった結果を組み立てる。"""
    return OrderResult(
        checked=False,
        skip_reason=reason,
        key_cols=list(key_cols),
        ambiguous_rows=ambiguous_rows,
        paired_rows=0,
        left_excluded=0,
        right_excluded=0,
        first_diff=None,
        diff_count=0,
        moved_rows=0,
        samples=pd.DataFrame(),
        left_cols=list(left_cols),
        right_cols=list(right_cols),
    )


def _skip_reason_text(
    paired_rows: int,
    left_total: int,
    right_total: int,
    ambiguous_rows: int,
) -> str:
    """行の並び順を判定できなかった理由を、件数つきで組み立てる。

    並び順の結果は出力の最後に出るため、理由に件数を埋めて
    上の行差分ブロックまで戻らなくても読めるようにする。

    出すのは観測された事実だけに留める。

    キーが対応しない理由は、文字化け、型の食い違い、前ゼロの脱落、
    空白、そもそも別データ、といくらでもある。
    ヒューリスティックで1つに決め打つと、外したときに
    調査を明後日の方向へ誘導するため、解釈は人間に渡す。
    """
    # 1行も対応づかず、かつ両側にそれなりの行数がある状態
    #
    # キー列自体が文字化けしているとこうなる。
    #
    # 行数の下限は、片側が空のファイルや数行しかないファイルが
    # この分岐へ落ちるのを防ぐ。どちらも数式の上では成立するが、
    # 突合が壊れている証拠にはならない。
    #
    # 小さいほうの行数で見るのは、
    # golden 1000行 / target 3行のような食い違いを
    # 「突合が壊れている」ではなく件数の提示へ回すため。
    if (
        paired_rows == 0
        and min(left_total, right_total)
        >= ORDER_ALL_UNMATCHED_MIN_ROWS
    ):
        return (
            "キーが1件も対応していないため: "
            f"{LEFT_KEY} {left_total:,}行 / "
            f"{RIGHT_KEY} {right_total:,}行、"
            "キー列の指定またはencodingを確認"
        )

    # 曖昧で落とした行があるなら、対応が少ない理由がそこにある
    # 件数だけ出すと、なぜ0行なのかが読み取れない
    ambiguous_text = (
        f"、うち同一キーで両側に値差分のため判定できない行 "
        f"{ambiguous_rows:,}行"
        if ambiguous_rows
        else ""
    )

    return (
        "対応する行が少なすぎるため: "
        f"対応{paired_rows:,}行 "
        f"({LEFT_KEY} {left_total:,}行 / "
        f"{RIGHT_KEY} {right_total:,}行"
        f"{ambiguous_text})"
    )


def _compare_order(
    left: pd.DataFrame,
    right: pd.DataFrame,
    common_cols: list[str],
    *,
    key_cols: list[str],
    left_cols: list[str],
    right_cols: list[str],
) -> OrderResult:
    """左右の行をペアにして、対応づいた行の相対順序を判定する。

    片側にしかない行があると位置は必ずズレるため、
    元の位置に居るかどうかでは判定できない。

    そこでペアになった行だけを取り出し、
    target側の位置が単調増加しているかを見る。
    行が抜けても増えても、残りの相対順序は判定できる。

    左右の行数が同じで全行がペアになる場合、
    単調増加であることと恒等写像であることは同値になる。
    つまり以前の判定はこの判定の特殊ケースにあたる。

    diff_countは「動いた行数」ではなく「相対位置が変わった行の数」。

    1行が先頭から100行目へ移動しただけでも、間の行がすべて
    1つずつ前へ詰まるため、101箇所という数え方になる。

    実際に動いた行数はmoved_rowsに入れる。
    最長増加部分列から外れた行、つまり
    それだけ動かせば順序が揃う最小の行数がこれにあたる。
    """
    left_total = len(left)
    right_total = len(right)

    left_text = _order_text(left, common_cols)
    right_text = _order_text(right, common_cols)

    mapped, ambiguous_rows = _pair_positions(
        left_text,
        right_text,
        key_cols,
    )

    paired = np.flatnonzero(mapped >= 0)

    if paired.size < ORDER_MIN_PAIRED_ROWS:
        return _skip_order(
            _skip_reason_text(
                int(paired.size),
                left_total,
                right_total,
                ambiguous_rows,
            ),
            key_cols,
            left_cols,
            right_cols,
            ambiguous_rows,
        )

    targets = mapped[paired]

    # 対応づいた行の中での順位
    #
    # 単調増加であることと、順位が0,1,2...と並ぶことは同値。
    #
    # 位置そのものではなく順位で見るのは、除外した行のぶんだけ
    # 位置がずれるため。全行がペアなら順位は位置と一致するので、
    # 除外がない場合の数え方は以前と変わらない。
    rank = np.argsort(np.argsort(targets))
    displaced = rank != np.arange(paired.size)

    diff_count = int(displaced.sum())

    first_diff = (
        int(paired[np.argmax(displaced)]) + 1
        if diff_count
        else None
    )

    # 実際に動いた行 = 全体 - 最長増加部分列
    keep = _longest_increasing(targets)
    moved_mask = np.ones(paired.size, dtype=bool)
    moved_mask[keep] = False

    moved_positions = paired[moved_mask][:ORDER_SAMPLE_CAP]

    # 行を識別する列
    # キーがあればキー列、なければ共通列すべて
    label_cols = list(key_cols) if key_cols else list(common_cols)
    label_header = "キー" if key_cols else "内容"

    samples = pd.DataFrame(
        {
            f"{LEFT_KEY}行": moved_positions + 1,
            f"{RIGHT_KEY}行": mapped[moved_positions] + 1,
            label_header: _order_labels(
                left_text,
                label_cols,
                moved_positions,
            ),
        }
    )

    return OrderResult(
        checked=True,
        skip_reason="",
        key_cols=list(key_cols),
        ambiguous_rows=ambiguous_rows,
        paired_rows=int(paired.size),
        left_excluded=left_total - int(paired.size),
        right_excluded=right_total - int(paired.size),
        first_diff=first_diff,
        diff_count=diff_count,
        moved_rows=int(moved_mask.sum()),
        samples=samples,
        left_cols=list(left_cols),
        right_cols=list(right_cols),
    )


# ────────────────────────────────────────────────────────────────────
# 結果の入れ物

@dataclass(frozen=True)
class VerifyResult:
    """goldenとtargetの突合結果。"""

    only_left: pd.DataFrame
    only_right: pd.DataFrame
    cell_diff: pd.DataFrame
    fuzzy_matched: pd.DataFrame
    only_left_cols: list[str]
    only_right_cols: list[str]
    order: OrderResult | None = None

    @property
    def is_match(self) -> bool:
        """値の差分がまったくない場合はTrueを返す。

        並び順は含めない。
        「値が一致」と「順序も一致」は要求レベルが別なので、
        並び順はis_order_matchで別に見る。
        """
        return (
            self.only_left.empty
            and self.only_right.empty
            and self.cell_diff.empty
            and not self.only_left_cols
            and not self.only_right_cols
        )

    @property
    def is_order_match(self) -> bool:
        """並び順が一致と判定された場合にTrueを返す。

        並び順を比較しなかった場合と、
        対応づいた行が足りず判定できなかった場合はFalseになる。

        片側にしかない行は比較から除外されるため、
        行が抜けていてもTrueになりうる。

        つまりこれは「残った行の相対順序が保たれているか」であって、
        行の過不足は見ていない。過不足はis_matchが見るので、
        両方を求めるなら呼び出し側で組み合わせる。

        何行を除外した上での判定かはorder.left_excluded /
        order.right_excludedに入る。
        """
        return (
            self.order is not None
            and self.order.is_match
        )


# ────────────────────────────────────────────────────────────────────
# 突合本体

def _verify(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_cols: list[str],
    float_atol: float = FLOAT_ATOL,
    *,
    check_order: bool = True,
) -> VerifyResult:
    """2つのDataFrameを突合し、列差分・行差分・セル差分を返す。

    key_colsが空リストの場合はキーなしモードになる。

    check_orderがTrueの場合、値の突合に加えて並び順も比較する。

    キーなしモードでは、完全一致する行を除外した後、
    golden側だけに残った行をonly_left、
    target側だけに残った行をonly_rightとして返す。

    キーがないため、残った行同士のセル差分までは判定しない。
    """
    print(f"\n🔖 {LEFT_KEY}:")

    for index, col in enumerate(left.columns, start=1):
        print(f"  {index:02d}:  {col}")

    print(f"\n🔖 {RIGHT_KEY}:")

    for index, col in enumerate(right.columns, start=1):
        print(f"  {index:02d}:  {col}")

    # 日付列はgolden側だけを基準に判定し、左右両方へ適用する
    date_cols = _resolve_date_cols(left)
    manual_date_cols = [col for col in DATE_COLS if col]

    print(f"\n🔖 date_cols (Total: {len(date_cols)}):")

    for index, col in enumerate(date_cols, start=1):
        source = "manual" if col in manual_date_cols else "auto"
        print(f"  {index:02d}:  {col} ({source})")

    # 並び順の比較には元ファイルの行順が要るため、ここでは並べ替えない
    left_u = _normalize(left, date_cols, sort_rows=False)
    right_u = _normalize(right, date_cols, sort_rows=False)

    # ────────────────────────────────────────────────────────────────
    # 列差分

    only_left_cols = [
        col
        for col in left_u.columns
        if col not in right_u.columns
    ]
    only_right_cols = [
        col
        for col in right_u.columns
        if col not in left_u.columns
    ]

    missing_keys = [
        col
        for col in key_cols
        if col in only_left_cols or col in only_right_cols
    ]

    if missing_keys:
        raise ValueError(
            "突合キー列が片側にしか存在しません: "
            f"{missing_keys}"
        )

    common_cols = [
        col
        for col in left_u.columns
        if col in right_u.columns
    ]

    if not common_cols:
        raise ValueError(
            "goldenとtargetに共通する列がありません"
        )

    # 共通列をtarget側の並びで持つ
    # common_colsはgolden側の並びなので、この2つを比べれば列の並び順が分かる
    right_col_order = [
        col
        for col in right_u.columns
        if col in left_u.columns
    ]

    left_u = left_u[common_cols]
    right_u = right_u[common_cols]

    # 共通列はすべてStage 1のmergeキーになるため、
    # merge前に左右のdtypeを揃えておく
    #
    # 日付列の判定はこの整合より前に済んでいる。
    # 判定をgolden基準から左右の和集合へ変えるなら、この呼び出しも判定より前へ移す。
    # 数値列にto_datetimeを当てるとエポックns扱いになり、警告なしで日付が壊れるため。
    left_u, right_u = _align_common_dtypes(
        left_u,
        right_u,
        common_cols,
    )

    # ここから先の突合は並び順に依存しないため、
    # 結果を実行ごとに安定させる決定的な順序へ並べ替える。
    # 並び順の比較には、並べ替える前のleft_u / right_uを使う。
    left_n = _sort_by_column_names(left_u)
    right_n = _sort_by_column_names(right_u)

    # 並び順の比較は行差分の結果に依存しない
    #
    # 対応づいた行だけの相対順序を見るので、
    # 片側にしかない行があっても残りは判定できる。
    # そのためキーあり・キーなしのどちらの経路でも同じ結果を使う。
    order = (
        _compare_order(
            left_u,
            right_u,
            common_cols,
            key_cols=key_cols,
            left_cols=common_cols,
            right_cols=right_col_order,
        )
        if check_order
        else None
    )

    # ────────────────────────────────────────────────────────────────
    # Stage 1: 完全一致行を並び順に依存せず吸収

    left_full = left_n.copy()
    right_full = right_n.copy()

    # cumcountは行のない組み合わせに何も返さないため、
    # observedはどちらでも結果が変わらない。
    # pandas 2の非推奨警告を避けて、pandas 3の既定値に合わせておく。
    left_full["_fseq"] = (
        left_full
        .groupby(common_cols, dropna=False, observed=True)
        .cumcount()
    )
    right_full["_fseq"] = (
        right_full
        .groupby(common_cols, dropna=False, observed=True)
        .cumcount()
    )

    full_match = left_full.merge(
        right_full,
        how="outer",
        on=common_cols + ["_fseq"],
        indicator=True,
    )

    resid_left = (
        full_match.loc[
            full_match["_merge"] == "left_only",
            common_cols,
        ]
        .reset_index(drop=True)
    )

    resid_right = (
        full_match.loc[
            full_match["_merge"] == "right_only",
            common_cols,
        ]
        .reset_index(drop=True)
    )

    # ────────────────────────────────────────────────────────────────
    # キーなしモード

    if not key_cols:
        (
            keyless_left,
            keyless_right,
            keyless_fuzzy,
        ) = _absorb_garbled_rows(
            resid_left,
            resid_right,
            common_cols,
        )

        return VerifyResult(
            only_left=keyless_left,
            only_right=keyless_right,
            cell_diff=pd.DataFrame(),
            fuzzy_matched=keyless_fuzzy,
            only_left_cols=only_left_cols,
            only_right_cols=only_right_cols,
            order=order,
        )

    # ────────────────────────────────────────────────────────────────
    # Stage 2: 完全一致しなかった残差だけをキーで対応づける

    resid_left = (
        resid_left
        .sort_values(sorted(common_cols))
        .reset_index(drop=True)
    )
    resid_right = (
        resid_right
        .sort_values(sorted(common_cols))
        .reset_index(drop=True)
    )

    resid_left["_seq"] = (
        resid_left
        .groupby(key_cols, dropna=False, observed=True)
        .cumcount()
    )
    resid_right["_seq"] = (
        resid_right
        .groupby(key_cols, dropna=False, observed=True)
        .cumcount()
    )

    merge_keys = key_cols + ["_seq"]

    key_match = resid_left[merge_keys].merge(
        resid_right[merge_keys],
        how="outer",
        on=merge_keys,
        indicator=True,
    )

    left_only_keys = key_match.loc[
        key_match["_merge"] == "left_only",
        merge_keys,
    ]
    right_only_keys = key_match.loc[
        key_match["_merge"] == "right_only",
        merge_keys,
    ]
    both_keys = key_match.loc[
        key_match["_merge"] == "both",
        merge_keys,
    ]

    only_left = (
        resid_left
        .merge(left_only_keys, on=merge_keys)
        .drop(columns="_seq")
    )

    only_right = (
        resid_right
        .merge(right_only_keys, on=merge_keys)
        .drop(columns="_seq")
    )

    # ────────────────────────────────────────────────────────────────
    # セル単位の差分

    # both_keysを左に置いて、左右の行順をこれに合わせる。
    #
    # 残差の並び替えはsorted(common_cols)、つまり列名のアルファベット順で
    # 行われるため、キー列より前に並ぶ列があると、そちらが第1ソートキーに
    # なる。残差行は左右で値が違うので、その場合キーの並びがズレる。
    #
    # 残差を左に置くとその並びがそのまま残り、left_bothとright_bothが
    # 同じキー集合を違う順序で持つことになって、セル比較が
    # 「Can only compare identically-labeled Series objects」で落ちる。
    left_both = (
        both_keys
        .merge(resid_left, on=merge_keys)
        .set_index(merge_keys)
    )

    right_both = (
        both_keys
        .merge(resid_right, on=merge_keys)
        .set_index(merge_keys)
    )

    # 左右の列順を統一
    right_both = right_both[left_both.columns]

    diffs: list[pd.DataFrame] = []
    fuzzy: list[pd.DataFrame] = []

    for col in left_both.columns:
        left_col = left_both[col]
        right_col = right_both[col]

        # 通常比較
        if (
            pd.api.types.is_numeric_dtype(left_col)
            and pd.api.types.is_numeric_dtype(right_col)
        ):
            mismatch = ~np.isclose(
                left_col,
                right_col,
                atol=float_atol,
                equal_nan=True,
            )
        else:
            mismatch = (
                left_col.astype(str)
                != right_col.astype(str)
            ).to_numpy()

        # 文字化け対象列はASCII英数字で再判定
        if mismatch.any() and col in GARBLED_COLS:
            left_signature = (
                left_col[mismatch]
                .map(_ascii_signature)
            )
            right_signature = (
                right_col[mismatch]
                .map(_ascii_signature)
            )

            still_mismatch = (
                left_signature != right_signature
            ).to_numpy()

            # 通常比較では不一致だが、
            # signature比較で一致したセルを記録
            rescued_mask = mismatch.copy()
            rescued_mask[mismatch] = ~still_mismatch

            if rescued_mask.any():
                fuzzy_frame = pd.DataFrame(
                    {
                        "column": col,
                        LEFT_KEY: left_col[rescued_mask],
                        RIGHT_KEY: right_col[rescued_mask],
                    }
                )
                fuzzy.append(fuzzy_frame.reset_index())

            final_mask = mismatch.copy()
            final_mask[mismatch] = still_mismatch
            mismatch = final_mask

        if mismatch.any():
            diff_frame = pd.DataFrame(
                {
                    "column": col,
                    LEFT_KEY: left_col[mismatch],
                    RIGHT_KEY: right_col[mismatch],
                }
            )
            diffs.append(diff_frame.reset_index())

    cell_diff = (
        pd.concat(diffs, ignore_index=True)
        if diffs
        else pd.DataFrame()
    )

    fuzzy_matched = (
        pd.concat(fuzzy, ignore_index=True)
        if fuzzy
        else pd.DataFrame()
    )

    if not cell_diff.empty:
        cell_diff = cell_diff.drop(columns="_seq")

    if not fuzzy_matched.empty:
        fuzzy_matched = fuzzy_matched.drop(columns="_seq")

    return VerifyResult(
        only_left=only_left,
        only_right=only_right,
        cell_diff=cell_diff,
        fuzzy_matched=fuzzy_matched,
        only_left_cols=only_left_cols,
        only_right_cols=only_right_cols,
        order=order,
    )


# ────────────────────────────────────────────────────────────────────
# ファイル読込

def _verify_files(
    golden_path: Path,
    target_path: Path,
    key_cols: list[str],
    *,
    check_order: bool = True,
) -> VerifyResult:
    """CSVファイルを読み込み、突合結果を返す。"""
    left = _read_csv(
        golden_path,
        "golden_path",
    )
    right = _read_csv(
        target_path,
        "target_path",
    )

    return _verify(
        left,
        right,
        key_cols=key_cols,
        check_order=check_order,
    )


# ────────────────────────────────────────────────────────────────────
# 結果表示

def _print_frame(
    df: pd.DataFrame,
    max_rows: int,
) -> None:
    """DataFrameの先頭を、1始まりの連番付きで表示する。

    max_rows行を超える場合は、切り詰めた旨の見出しを添える。
    """
    if len(df) > max_rows:
        print(f"  = top{max_rows} =")

    shown = (
        df
        .head(max_rows)
        .reset_index(drop=True)
    )
    shown.index += 1  # 表示の連番を1始まりにする

    print(shown.to_string())


def _print_order(
    order: OrderResult | None,
    max_rows: int,
) -> None:
    """並び順の比較結果をコンソールへ表示する。"""
    if order is None:
        print("\n➖ 並び順: 比較なし (--no-order)")
        return

    mark_ok = "✅"
    mark_ng = "⚠️"

    # 列の並び順
    col_pairs = [
        (index, left_col, right_col)
        for index, (left_col, right_col) in enumerate(
            zip(order.left_cols, order.right_cols, strict=True),
            start=1,
        )
        if left_col != right_col
    ]

    if not col_pairs:
        print(
            f"\n{mark_ok} 列の並び順: 一致 "
            f"(共通{len(order.left_cols)}列)"
        )
    else:
        print(
            f"\n{mark_ng} 列の並び順: 不一致 "
            f"(共通{len(order.left_cols)}列中 "
            f"{len(col_pairs)}箇所)"
        )

        _print_frame(
            pd.DataFrame(
                col_pairs,
                columns=["位置", LEFT_KEY, RIGHT_KEY],
            ),
            max_rows,
        )

    # 行の並び順
    if not order.checked:
        print(
            f"\n➖ 行の並び順: 判定なし "
            f"({order.skip_reason})"
        )
        return

    # 除外した行があるなら「一致」と裸で言わない
    #
    # 何行を除外した上での話なのかを必ず並べる。
    # ここを省くと「順序OK」で行差分そのものを見落とす。
    excluded_note = (
        f"  ※ 片側だけの行 {LEFT_KEY} {order.left_excluded:,}行・"
        f"{RIGHT_KEY} {order.right_excluded:,}行 は比較から除外"
    )

    # キーなしモードは行を対応づける手段がないため、
    # 値が1セル違うだけの行も除外側へ落ちる。
    # 行の欠落と読み違えられないよう添える。
    keyless_note = (
        "  ※ キーなしモードでは、"
        "値が1セル違う行も除外側に入る"
    )

    # 除外の内訳のうち、行の過不足ではなく判定不能によるもの
    # 行差分ブロックを見ても出てこない数字なので、ここで説明する
    ambiguous_note = (
        "  ※ うち同一キーで両側に値差分のある "
        f"{order.ambiguous_rows:,}行 は判定できないため除外"
    )

    if order.diff_count == 0:
        if order.has_excluded:
            print(
                f"\n{mark_ok} 行の並び順: 相対順序は一致 "
                f"(対応{order.paired_rows:,}行 / "
                f"除外 {LEFT_KEY} {order.left_excluded:,}行・"
                f"{RIGHT_KEY} {order.right_excluded:,}行)"
            )

            if not order.key_cols:
                print(keyless_note)
        else:
            print(
                f"\n{mark_ok} 行の並び順: 一致 "
                f"({order.paired_rows:,}行)"
            )

        if order.has_ambiguity:
            print(ambiguous_note)

        return

    print(
        f"\n{mark_ng} 行の並び順: 不一致 "
        f"(対応{order.paired_rows:,}行中 "
        f"{order.first_diff:,}行目から {order.diff_count:,}箇所 / "
        f"動いた行 {order.moved_rows:,}行)"
    )

    # 1行動いただけでも以降が全部ズレるため、
    # 箇所数を「動いた行数」と読み違えないよう毎回添える
    print(
        "  ※ 箇所数は相対位置が変わった行の数であって、"
        "動いた行数ではない"
    )
    print(
        "  ※ 対応づけ: 全列一致"
        + (
            " → キー一致 (" + ", ".join(order.key_cols) + ")"
            if order.key_cols
            else ""
        )
    )

    if order.has_excluded:
        print(excluded_note)

        if not order.key_cols:
            print(keyless_note)

    if order.has_ambiguity:
        print(ambiguous_note)

    # 注記が続いた直後に表が来ると、注記が表の見出しに見える
    print()
    print("  = 動いた行 =")

    _print_frame(order.samples, max_rows)


def _print_result(
    result: VerifyResult,
    key_cols: list[str],
    *,
    max_rows: int,
) -> None:
    """突合結果をコンソールへ表示する。"""
    mark_ok = "✅"
    mark_ng = "⚠️"

    mark = mark_ok if not result.only_left_cols else mark_ng
    print(
        f"\n{mark} {LEFT_KEY} のみの列: "
        f"{len(result.only_left_cols)}列"
    )

    for index, col in enumerate(
        result.only_left_cols,
        start=1,
    ):
        print(f"  {index}:  {col}")

    mark = mark_ok if not result.only_right_cols else mark_ng
    print(
        f"\n{mark} {RIGHT_KEY} のみの列: "
        f"{len(result.only_right_cols)}列"
    )

    for index, col in enumerate(
        result.only_right_cols,
        start=1,
    ):
        print(f"  {index}:  {col}")

    # goldenにだけある行
    mark = mark_ok if result.only_left.empty else mark_ng
    print(
        f"\n{mark} {LEFT_KEY} のみの行: "
        f"{len(result.only_left)}行"
    )

    if not result.only_left.empty:
        display_cols = [
            col
            for col in key_cols
            if col in result.only_left.columns
        ]

        subset = (
            result.only_left[display_cols]
            if display_cols
            else result.only_left
        )

        _print_frame(subset, max_rows)

    # targetにだけある行
    mark = mark_ok if result.only_right.empty else mark_ng
    print(
        f"\n{mark} {RIGHT_KEY} のみの行: "
        f"{len(result.only_right)}行"
    )

    if not result.only_right.empty:
        display_cols = list(
            dict.fromkeys(
                key_cols
                + [col for col in EXTRA_COLS if col]
            )
        )

        display_cols = [
            col
            for col in display_cols
            if col in result.only_right.columns
        ]

        subset = (
            result.only_right[display_cols]
            if display_cols
            else result.only_right
        )

        _print_frame(subset, max_rows)

    # セル差分
    mark = mark_ok if result.cell_diff.empty else mark_ng
    print(
        f"\n{mark} 両方にあるが値が違う行: "
        f"{len(result.cell_diff)}行"
    )

    if not result.cell_diff.empty:
        _print_frame(result.cell_diff, max_rows)

    # 文字化けと思われる差分
    unique_pairs = pd.DataFrame()

    if not result.fuzzy_matched.empty:
        unique_pairs = (
            result.fuzzy_matched[
                ["column", LEFT_KEY, RIGHT_KEY]
            ]
            .drop_duplicates()
            .sort_values(["column", LEFT_KEY])
            .reset_index(drop=True)
        )
        unique_pairs.index += 1

    mark = (
        mark_ok
        if result.fuzzy_matched.empty
        else mark_ng
    )

    print(
        f"\n{mark} 文字化けだと思われるもの: "
        f"{len(result.fuzzy_matched)}行, "
        f"{len(unique_pairs)}件"
    )

    if not result.fuzzy_matched.empty:
        print(unique_pairs)

    _print_order(result.order, max_rows)


# ────────────────────────────────────────────────────────────────────
# 公開関数

def run_verify(
    golden_name: str | None = None,
    key_cols: list[str] | None = None,
    *,
    target_path: Path = DEFAULT_TARGET,
    max_rows: int = DEFAULT_MAX_ROWS,
    check_order: bool = True,
) -> VerifyResult:
    """goldenとtargetを突合し、結果を表示して返す。

    Parameters
    ----------
    golden_name:
        golden CSVのstem。

        DEFAULT_GOLDEN_DIR配下から、この名前の.csvを探す。

        Noneまたは空文字の場合はDEFAULT_GOLDEN_NAMEを使用する。

    key_cols:
        突合キー列。

        None:
            DEFAULT_KEYSを使用する。

        []:
            キーなしモードで比較する。

        列名のリスト:
            指定した列をキーとして比較する。

    target_path:
        target CSVのパス。

        デフォルトはoutput/debug/_test.csv。

    max_rows:
        差分の表示行数。

        省略時はDEFAULT_MAX_ROWSを使用する。

    check_order:
        並び順まで比較するかどうか。

        Trueの場合、値の突合に加えて行と列の並び順を比較する。

        行の並び順は、片側にしかない行がない場合だけ判定できる。
        判定結果はVerifyResult.orderとis_order_matchで参照する。
    """
    # head(-n)は末尾n行を落とす意味になり、
    # 見出しと実際の表示行数が食い違うため先に弾く
    if max_rows < 1:
        raise ValueError(
            f"max_rowsは1以上を指定してください: {max_rows}"
        )

    actual_keys = (
        DEFAULT_KEYS.copy()
        if key_cols is None
        else list(key_cols)
    )

    golden_path = _resolve_golden_path(golden_name)

    print(f"-- RUN --\n\n🎈 {__name__}\n[golden] {golden_path}")
    print(f"target : {target_path}")
    print(
        "key    : "
        + (
            ", ".join(actual_keys)
            if actual_keys
            else "(no key)"
        )
    )
    print(
        "order  : "
        + ("比較する" if check_order else "比較しない")
    )

    result = _verify_files(
        golden_path=golden_path,
        target_path=target_path,
        key_cols=actual_keys,
        check_order=check_order,
    )

    _print_result(
        result,
        key_cols=actual_keys,
        max_rows=max_rows,
    )

    # 差分が長いと冒頭の見出しまで戻らないと確認できないため、
    # どのファイルを比べた結果なのかを末尾にもう一度出す
    print(f"\n[golden] {golden_path}")
    print(f"target : {target_path}")

    return result


def main() -> None:
    """コマンドライン引数を受け取り、突合を実行する。"""
    parser = argparse.ArgumentParser(
        description="golden出力とtarget出力の突合"
    )

    parser.add_argument(
        "golden_name",
        nargs="?",
        default=None,
        help=(
            "golden CSVのstem。"
            f"省略時は{DEFAULT_GOLDEN_NAME}"
        ),
    )

    parser.add_argument(
        "target",
        nargs="?",
        default=DEFAULT_TARGET,
        type=Path,
        help=(
            "target出力CSV。"
            f"省略時は{DEFAULT_TARGET}"
        ),
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            "差分の表示行数。"
            f"省略時は{DEFAULT_MAX_ROWS}"
        ),
    )

    parser.add_argument(
        "--no-order",
        action="store_true",
        help=(
            "並び順の比較をスキップする。"
            "省略時は値の突合に続けて並び順も比較する"
        ),
    )

    parser.add_argument(
        "--key",
        nargs="*",
        default=None,
        help=(
            "突合キー列。"
            "オプション自体を省略するとDEFAULT_KEYS、"
            "--keyだけを指定するとキーなし"
        ),
    )

    args = parser.parse_args()

    run_verify(
        golden_name=args.golden_name,
        key_cols=args.key,
        target_path=args.target,
        max_rows=args.max_rows,
        check_order=not args.no_order,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), end="")
        print("\n❌ Exit code: 1")
        raise SystemExit(1)
    else:
        print("\n✅ Exit code: 0")
