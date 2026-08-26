"""2つのCSV出力を突合するスクリプト。

基準となる正解データをgolden、検証対象の出力をtargetと呼ぶ。

移行前後の出力比較や、リファクタ前後の回帰確認などに使う。
"""

from __future__ import annotations

import argparse
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
            df = pd.read_csv(path, encoding=encoding)
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
    for col in _text_cols(normalized):
        normalized[col] = normalized[col].str.strip()

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


@dataclass(frozen=True)
class OrderResult:
    """並び順の比較結果。

    行の並び順は、左右の行の集合が一致しているときしか判定できない。
    判定できなかった場合はcheckedがFalseになり、skip_reasonに理由が入る。

    列の並び順は前提条件なしで判定できるため、checkedとは無関係に埋まる。
    """

    checked: bool
    skip_reason: str
    compare_cols: list[str]
    key_dup_rows: int
    row_total: int
    first_diff: int | None
    diff_count: int
    samples: pd.DataFrame
    left_cols: list[str]
    right_cols: list[str]

    @property
    def key_duplicated(self) -> bool:
        """比較に使ったキー列に重複がある場合はTrueを返す。

        重複がある場合、同一キー内での行の入れ替わりは検出できない。
        row_matchがTrueでも「確認できた範囲では一致」の意味になる。
        """
        return self.key_dup_rows > 0

    @property
    def row_match(self) -> bool:
        """行の並び順が一致していると確認できた場合のみTrueを返す。

        キー列に重複がある場合、同一キー内の入れ替わりは見えないため、
        Trueでも「キー列で確認できる範囲では一致」に留まる。
        """
        return self.checked and self.diff_count == 0

    @property
    def col_match(self) -> bool:
        """共通列の並び順が一致している場合はTrueを返す。"""
        return self.left_cols == self.right_cols

    @property
    def is_match(self) -> bool:
        """行と列の並び順が両方とも一致している場合のみTrueを返す。"""
        return self.row_match and self.col_match


def _order_text(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """並び順の比較に使う列を、左右で表記が揃う文字列へ変換する。

    文字化け対象列はASCII英数字のsignatureへ置き換える。

    _absorb_garbled_rowsは文字化けした行をsignatureでペアにして
    差分から取り除くが、生の値は左右で違ったまま残る。

    そのまま位置比較をすると、吸収したはずの行で
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
    positions: np.ndarray,
) -> list[str]:
    """サンプル表示用に、比較対象列の値を1行1文字列へまとめる。

    キーなしモードでは共通列すべてが比較対象になり、
    そのまま並べると1行が横に伸びすぎるため頭で切る。
    """
    labels: list[str] = []

    for row in text.to_numpy()[positions]:
        label = " | ".join(row)

        if len(label) > ORDER_LABEL_WIDTH:
            label = label[: ORDER_LABEL_WIDTH - 1] + "…"

        labels.append(label)

    return labels


def _skip_order(
    reason: str,
    compare_cols: list[str],
    left_cols: list[str],
    right_cols: list[str],
    key_dup_rows: int,
) -> OrderResult:
    """行の並び順を判定しなかった結果を組み立てる。"""
    return OrderResult(
        checked=False,
        skip_reason=reason,
        compare_cols=list(compare_cols),
        key_dup_rows=key_dup_rows,
        row_total=0,
        first_diff=None,
        diff_count=0,
        samples=pd.DataFrame(),
        left_cols=list(left_cols),
        right_cols=list(right_cols),
    )


def _compare_order(
    left: pd.DataFrame,
    right: pd.DataFrame,
    compare_cols: list[str],
    *,
    left_cols: list[str],
    right_cols: list[str],
    key_dup_rows: int,
) -> OrderResult:
    """左右の行を先頭から突き合わせ、位置がズレた箇所を数える。

    diff_countは「動いた行数」ではなく「位置がズレた箇所の数」。

    1行が先頭から100行目へ移動しただけでも、間の行がすべて
    1つずつ前へ詰まるため、101箇所という数え方になる。

    ここを行数として読むと規模を大きく誤るため、
    表示側のラベルも「箇所」で統一している。

    実際に動いた行数を出すには順列のランク比較が必要になるが、
    最初のズレ位置さえ分かれば目視で追えるため、そこまではやらない。
    """
    if len(left) != len(right):
        return _skip_order(
            "行数が一致していないため",
            compare_cols,
            left_cols,
            right_cols,
            key_dup_rows,
        )

    left_text = _order_text(left, compare_cols)
    right_text = _order_text(right, compare_cols)

    mismatch = (
        left_text.to_numpy() != right_text.to_numpy()
    ).any(axis=1)

    diff_count = int(mismatch.sum())

    first_diff = (
        int(np.argmax(mismatch)) + 1
        if diff_count
        else None
    )

    positions = np.flatnonzero(mismatch)[:ORDER_SAMPLE_CAP]

    samples = pd.DataFrame(
        {
            "位置": positions + 1,
            LEFT_KEY: _order_labels(left_text, positions),
            RIGHT_KEY: _order_labels(right_text, positions),
        }
    )

    return OrderResult(
        checked=True,
        skip_reason="",
        compare_cols=list(compare_cols),
        key_dup_rows=key_dup_rows,
        row_total=len(left),
        first_diff=first_diff,
        diff_count=diff_count,
        samples=samples,
        left_cols=list(left_cols),
        right_cols=list(right_cols),
    )


def _skip_reason_text(
    only_left: pd.DataFrame,
    only_right: pd.DataFrame,
    *,
    keyed: bool,
    left_total: int,
    right_total: int,
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
    left_n = len(only_left)
    right_n = len(only_right)

    # キーなしモードは行を対応づける手段がないため、
    # 値が1セル違うだけの行も、行そのものの欠落や余剰も、
    # 区別されないまま同じ「片側だけの行」に落ちる。
    #
    # どちらが起きたかは判定していないので、
    # 片方に決め打たず、モードの制約だけを添える。
    if not keyed:
        return (
            "片側だけの行があるため: "
            f"{LEFT_KEY}のみ{left_n:,}行 / "
            f"{RIGHT_KEY}のみ{right_n:,}行、"
            "キーなしモードでは値が1セル違う行も片側だけになる"
        )

    # 残差ではなく全行が片側だけに落ちている状態
    #
    # キー列自体が文字化けしているとこうなる。
    # 片側が空の場合も数式の上では成立してしまうため、
    # 両側に行があることを条件に入れる。
    if (
        left_total > 0
        and right_total > 0
        and left_n == left_total
        and right_n == right_total
    ):
        return (
            "キーが1件も対応していないため: "
            f"{LEFT_KEY} {left_total:,}行 / "
            f"{RIGHT_KEY} {right_total:,}行、"
            "キー列の指定またはencodingを確認"
        )

    if right_n == 0:
        return (
            f"{LEFT_KEY}にしかないキーが"
            f"{left_n:,}行あるため"
        )

    if left_n == 0:
        return (
            f"{RIGHT_KEY}にしかないキーが"
            f"{right_n:,}行あるため"
        )

    return (
        "キーが対応しない行があるため: "
        f"{LEFT_KEY}のみ{left_n:,}行 / "
        f"{RIGHT_KEY}のみ{right_n:,}行"
    )


def _resolve_order(
    left: pd.DataFrame,
    right: pd.DataFrame,
    compare_cols: list[str],
    *,
    left_cols: list[str],
    right_cols: list[str],
    key_dup_rows: int,
    keyed: bool,
    only_left: pd.DataFrame,
    only_right: pd.DataFrame,
) -> OrderResult:
    """行の集合が一致しているときだけ、行の並び順を比較する。

    片側にしかない行が残っている状態で位置比較をすると、
    欠落や余剰の位置から後ろがすべてズレて情報にならない。

    一方でセル差分(cell_diff)は前提条件に含めない。
    キーが対応してさえいれば、値が違っても並び順は正しく判定でき、
    「値は違うが順序は保たれている」を拾えるほうが情報量が多い。
    """
    if not (only_left.empty and only_right.empty):
        return _skip_order(
            _skip_reason_text(
                only_left,
                only_right,
                keyed=keyed,
                left_total=len(left),
                right_total=len(right),
            ),
            compare_cols,
            left_cols,
            right_cols,
            key_dup_rows,
        )

    return _compare_order(
        left,
        right,
        compare_cols,
        left_cols=left_cols,
        right_cols=right_cols,
        key_dup_rows=key_dup_rows,
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
        """並び順が一致していると確認できた場合のみTrueを返す。

        並び順を比較しなかった場合と、
        行セットが違って判定できなかった場合はFalseになる。

        両方を満たすことを求めるなら、
        呼び出し側でis_matchと組み合わせる。
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

    # 並び順の比較に使う列
    #
    # キーありモードはキー列だけでよい。
    # キーが対応していれば値が違っても順序は判定できるので、
    # 全列で比べるとセル差分のある行まで「並び順が違う」に化ける。
    #
    # キーなしモードは行を識別できる列が他にないため共通列すべてを使う。
    order_cols = (
        list(key_cols)
        if key_cols
        else list(common_cols)
    )

    # キー列に重複があると、同一キー内での行の入れ替わりは
    # order_colsの比較に現れない
    #
    #   golden: ID=[1, 1, 2], V=[a, b, c]
    #   target: ID=[1, 1, 2], V=[b, a, c]
    #
    # Stage 1が値差分を吸収し、キー列だけを見ると1,1,2どうしで
    # 一致するため、実際は入れ替わっているのに「一致」と出る。
    #
    # order_colsを共通列すべてへ広げれば見えるようになるが、
    # 今度はセル差分のある行が「並び順が違う」に化けて、
    # 値の一致と順序の一致を分けた意味がなくなる。
    #
    # そのため判定は変えず、確認できた範囲を表示側で明示する。
    #
    # 判定できる場合は左右のキーが同じ多重集合になっているため、
    # 重複の有無はgolden側だけ見れば足りる。
    key_dup_rows = (
        int(
            left_u
            .duplicated(subset=key_cols, keep=False)
            .sum()
        )
        if key_cols
        else 0
    )

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

        keyless_order = (
            _resolve_order(
                left_u,
                right_u,
                order_cols,
                left_cols=common_cols,
                right_cols=right_col_order,
                key_dup_rows=key_dup_rows,
                keyed=bool(key_cols),
                only_left=keyless_left,
                only_right=keyless_right,
            )
            if check_order
            else None
        )

        return VerifyResult(
            only_left=keyless_left,
            only_right=keyless_right,
            cell_diff=pd.DataFrame(),
            fuzzy_matched=keyless_fuzzy,
            only_left_cols=only_left_cols,
            only_right_cols=only_right_cols,
            order=keyless_order,
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

    order = (
        _resolve_order(
            left_u,
            right_u,
            order_cols,
            left_cols=common_cols,
            right_cols=right_col_order,
            key_dup_rows=key_dup_rows,
            keyed=bool(key_cols),
            only_left=only_left,
            only_right=only_right,
        )
        if check_order
        else None
    )

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

    # キー列に重複があると同一キー内の入れ替わりが見えないため、
    # 「一致」と言い切らず、確認できた範囲を添える
    dup_note = (
        "  ※ キー列に重複あり "
        f"({order.key_dup_rows:,}行)。"
        "同一キー内の入れ替わりは検出できない"
    )

    if order.diff_count == 0:
        print(
            f"\n{mark_ok} 行の並び順: 一致 "
            f"({order.row_total:,}行)"
        )

        if order.key_duplicated:
            print(dup_note)

        return

    print(
        f"\n{mark_ng} 行の並び順: 不一致 "
        f"({order.row_total:,}行中 "
        f"{order.first_diff:,}行目から {order.diff_count:,}箇所)"
    )

    # 1行動いただけでも以降が全部ズレるため、
    # 箇所数を「動いた行数」と読み違えないよう毎回添える
    print(
        "  ※ 箇所数は位置がズレた行位置の数であって、"
        "動いた行数ではない"
    )
    print(
        "  ※ 比較列: "
        + ", ".join(order.compare_cols)
    )

    if order.key_duplicated:
        print(dup_note)

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
