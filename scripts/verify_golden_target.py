"""Alteryx出力（golden）とpandas出力（target）の突合スクリプト。"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# デフォルト設定

DATA_DIR = Path(__file__).resolve().parents[1]

DEFAULT_GOLDEN_NAME = "93"
DEFAULT_GOLDEN_DIR = DATA_DIR / "sample" / "50"
DEFAULT_TARGET = DATA_DIR / "output" / "debug" / "_test.csv"

DEFAULT_KEYS: list[str] = ["EL_ID"]


# ────────────────────────────────────────────────────────────────────
# ファイル固有の設定

# 文字化けしているカラム
# 文字化けによるdiff検出を避けたい場合は、該当カラムを指定する
GARBLED_COLS: list[str] = [
    "インドア局名(正式名称)",
    "対応階数",
]

# 日付型カラム
# mm/dd と m/d の表示違いによるdiff検出を避けたい場合に指定する
# 自動検出で拾えないカラムだけを手動指定する
DATE_COLS: list[str] = []

# only_right表示時に追加するカラム
EXTRA_COLS: list[str] = []


# ────────────────────────────────────────────────────────────────────
# 比較設定

LEFT_KEY = "Alteryx"
RIGHT_KEY = "Python"

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
# 日付列の自動判定

def _detect_date_cols(
    df: pd.DataFrame,
    min_ratio: float = 0.95,
    sample: int = 2000,
) -> list[str]:
    """object型カラムのうち、値の大半が日付として解釈できる列名を返す。

    左右で個別に判定すると非対称になり、偽差分の原因になるため、
    golden側だけに対して呼び出す。

    純粋な数値列は、IDやコードを日付と誤判定しないよう除外する。
    """
    cols: list[str] = []

    for col in df.select_dtypes(include="object").columns:
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

    return DEFAULT_GOLDEN_DIR / f"{stem}.csv"


# ────────────────────────────────────────────────────────────────────
# 正規化

def _normalize(
    df: pd.DataFrame,
    date_cols: list[str],
) -> pd.DataFrame:
    """突合前のDataFrameを正規化する。"""
    normalized = df.copy()

    # 文字列カラムの前後空白を除去
    for col in normalized.select_dtypes(include="object").columns:
        normalized[col] = normalized[col].str.strip()

    # NaNと空文字を統一
    # 数値列まで空文字で埋めるとobject型になり、
    # np.iscloseによる数値比較が効かなくなるため対象外とする
    object_cols = normalized.select_dtypes(include="object").columns
    normalized[object_cols] = normalized[object_cols].fillna("")

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
    object_cols = normalized.select_dtypes(include="object").columns
    normalized[object_cols] = normalized[object_cols].fillna("")

    # 列名順による決定的な並び替え
    if len(normalized.columns) > 0:
        normalized = normalized.sort_values(
            sorted(normalized.columns)
        )

    return normalized.reset_index(drop=True)


def _ascii_signature(value: str) -> str:
    """文字列からASCII英数字だけを抽出したsignatureを返す。"""
    normalized = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    return "".join(ALNUM_PATTERN.findall(normalized))


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
        .groupby(common_cols, dropna=False)
        .cumcount()
    )
    right_sig["_gseq"] = (
        right_sig
        .groupby(common_cols, dropna=False)
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

    @property
    def is_match(self) -> bool:
        """差分がまったくない場合はTrueを返す。"""
        return (
            self.only_left.empty
            and self.only_right.empty
            and self.cell_diff.empty
            and not self.only_left_cols
            and not self.only_right_cols
        )


# ────────────────────────────────────────────────────────────────────
# 突合本体

def _verify(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_cols: list[str],
    float_atol: float = FLOAT_ATOL,
) -> VerifyResult:
    """2つのDataFrameを突合し、列差分・行差分・セル差分を返す。

    key_colsが空リストの場合はキーなしモードになる。

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

    left_n = _normalize(left, date_cols)
    right_n = _normalize(right, date_cols)

    # ────────────────────────────────────────────────────────────────
    # 列差分

    only_left_cols = [
        col
        for col in left_n.columns
        if col not in right_n.columns
    ]
    only_right_cols = [
        col
        for col in right_n.columns
        if col not in left_n.columns
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
        for col in left_n.columns
        if col in right_n.columns
    ]

    if not common_cols:
        raise ValueError(
            "goldenとtargetに共通する列がありません"
        )

    left_n = left_n[common_cols]
    right_n = right_n[common_cols]

    # ────────────────────────────────────────────────────────────────
    # Stage 1: 完全一致行を並び順に依存せず吸収

    left_full = left_n.copy()
    right_full = right_n.copy()

    left_full["_fseq"] = (
        left_full
        .groupby(common_cols, dropna=False)
        .cumcount()
    )
    right_full["_fseq"] = (
        right_full
        .groupby(common_cols, dropna=False)
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
        .groupby(key_cols, dropna=False)
        .cumcount()
    )
    resid_right["_seq"] = (
        resid_right
        .groupby(key_cols, dropna=False)
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

    left_both = (
        resid_left
        .merge(both_keys, on=merge_keys)
        .set_index(merge_keys)
    )

    right_both = (
        resid_right
        .merge(both_keys, on=merge_keys)
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
    )


# ────────────────────────────────────────────────────────────────────
# ファイル読込

def _verify_files(
    golden_path: Path,
    target_path: Path,
    key_cols: list[str],
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
    )


# ────────────────────────────────────────────────────────────────────
# 結果表示

def _print_result(
    result: VerifyResult,
    key_cols: list[str],
    *,
    max_rows: int = 20,
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

        if len(subset) > max_rows:
            print(f"  = top{max_rows} =")

        print(
            subset
            .head(max_rows)
            .to_string()
        )

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

        if len(subset) > max_rows:
            print(f"  = top{max_rows} =")

        print(
            subset
            .head(max_rows)
            .to_string()
        )

    # セル差分
    mark = mark_ok if result.cell_diff.empty else mark_ng
    print(
        f"\n{mark} 両方にあるが値が違う行: "
        f"{len(result.cell_diff)}行"
    )

    if not result.cell_diff.empty:
        if len(result.cell_diff) > max_rows:
            print(f"  = top{max_rows} =")

        shown = (
            result.cell_diff
            .head(max_rows)
            .reset_index(drop=True)
        )
        shown.index += 1  # 表示の連番を1始まりにする

        print(shown.to_string())

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


# ────────────────────────────────────────────────────────────────────
# 公開関数

def run_verify(
    golden_name: str | None = None,
    key_cols: list[str] | None = None,
    *,
    target_path: Path = DEFAULT_TARGET,
) -> VerifyResult:
    """goldenとtargetを突合し、結果を表示して返す。

    Parameters
    ----------
    golden_name:
        golden CSVのstem。

        Noneまたは空文字の場合はDEFAULT_GOLDEN_NAMEを使用する。

        例:
            "93"

    key_cols:
        突合キー列。

        None:
            DEFAULT_KEYSを使用する。

        []:
            キーなしモードで比較する。

        ["EL_ID"]:
            EL_IDをキーとして比較する。

    target_path:
        target CSVのパス。

        デフォルトはoutput/debug/_test.csv。
    """
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

    result = _verify_files(
        golden_path=golden_path,
        target_path=target_path,
        key_cols=actual_keys,
    )

    _print_result(
        result,
        key_cols=actual_keys,
    )

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
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "予期しないエラーが発生しました"
        )
        print(
            "\n❌ Exit code: 1",
            file=sys.stderr,
        )
        raise SystemExit(1)
    else:
        print("\n✅ Exit code: 0")
