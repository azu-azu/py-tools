"""golden 出力と target 出力の突合スクリプト"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# CSV　都度変更する
TARGET_CSV_NAME = "" + ".csv"
DEFAULT_KEYS: list[str] = [""]

# 文字化けしてるカラム名　都度変更する
GARBLED_COLS: list[str] = []

# 日付型カラム
DATE_COLS: list[str] = [""]

# only_right 表示時に追加する列（_L/_R 展開前の名前）　都度変更する
EXTRA_COLS: list[str] = [""]

DATA_DIR = Path(__file__).resolve().parents[1]  # py-tools プロジェクト直下
DEFAULT_GOLDEN = DATA_DIR / TARGET_CSV_NAME
DEFAULT_TARGET = DATA_DIR / "output" / "debug" / "_test.csv"

LEFT_KEY = "left"
RIGHT_KEY = "right"

FLOAT_ATOL: float = 1e-9

ASCII_PATTERN = re.compile(r"[\x20-\x7e]+")
ALNUM_PATTERN = re.compile(r"[0-9A-Za-z]+")

DEFAULT_ENCODINGS: tuple[str, ...] = (
    "cp932",
    "shift_jis_2004",
    "utf-8-sig",
)


# ────────────────────────────────────────────────────────────────────
# CSV読み込み

def read_csv(path: Path, label: str) -> pd.DataFrame:
    """encoding を順に試して CSV を読む。全滅したら最後の encoding で強制オープン（warning 付き）。"""
    for enc in DEFAULT_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✅ {label}: encoding={enc}")
            return df
        except (UnicodeDecodeError, LookupError):
            print(f"⚠️ {label}: encoding={enc} NG")

    last = DEFAULT_ENCODINGS[-1]
    print(f"⚠️ {label}: 全 encoding 失敗、{last} で強制オープン（文字化けの可能性あり）")
    return pd.read_csv(path, encoding=last, encoding_errors="replace")


# ────────────────────────────────────────────────────────────────────
# 日付列の自動判定

def detect_date_cols(
    df: pd.DataFrame, min_ratio: float = 0.95, sample: int = 2000
) -> list[str]:
    """object 型カラムのうち、値の大半が日付としてパースできる列名を返す。

    突合ツールで左右バラバラに判定すると非対称になって偽差分の原因になるため、
    この関数は「基準側（golden）1つ」に対してだけ呼び、得たリストを両方に適用すること。

    - 純粋な数値列（ID・郵便番号・コード等）は誤検出を避けるため除外する
    - パース成功率が min_ratio 以上の列だけを日付列とみなす
    """
    cols: list[str] = []
    for col in df.select_dtypes(include="object").columns:
        s = df[col].astype(str).str.strip()
        s = s[s != ""]
        if s.empty:
            continue
        if s.str.fullmatch(r"\d+").all():  # 純粋な数字列（ID等）は日付扱いしない
            continue
        smp = s.sample(min(len(s), sample), random_state=0)
        # 混在フォーマットを許容した best-effort な判定なので、フォーマット推論の
        # UserWarning は抑止する（本判定は verify() 側のログで可視化される）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ratio = pd.to_datetime(smp, errors="coerce").notna().mean()
        if ratio >= min_ratio:
            cols.append(col)
    return cols


def resolve_date_cols(reference: pd.DataFrame) -> list[str]:
    """手動指定(DATE_COLS)と自動判定を統合した日付列リストを返す。

    手動指定を優先しつつ自動検出を補助として足す。順序は保ちつつ重複を除く。
    """
    manual = [c for c in DATE_COLS if c]  # DATE_COLS の空文字プレースホルダを除去
    auto = detect_date_cols(reference)
    return list(dict.fromkeys(manual + auto))


# ────────────────────────────────────────────────────────────────────
# 正規化

def normalize(
    df: pd.DataFrame, key_cols: list[str], date_cols: list[str]
) -> pd.DataFrame:
    """突合前の正規化。"""
    df = df.copy()

    # 文字列カラムの前後空白を除去
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()

    # NaN と空文字を統一（比較用）。数値カラムを対象外にするのは、
    # fillna("") で object 化すると verify() の数値許容比較(np.isclose)が効かなくなるため
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].fillna("")

    # 日付カラムを統一フォーマットに正規化
    for col in date_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")

            lost = df[col].notna() & df[col].ne("") & parsed.isna()

            # coerce によって握りつぶされたエラーがあった場合
            if lost.any():
                print(f"⚠️ {col}: 日付として解釈できない値 {lost.sum()}件")
                print(df.loc[lost, col].drop_duplicates().head(10).to_string())

            df[col] = parsed.dt.strftime("%Y-%m-%d")

    # NaN と空文字を再度統一（日付正規化で新たに出た NaN を吸収）
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].fillna("")

    # 行順の差を消す：全カラムでソートしてから、同一キー内の連番(_seq)を振る
    # (①key_colsに重複があっても total 行が一意に対応づく。②連番の前提を
    #  ファイルの元の並び順に依存させないため、全カラムでの決定的整列が必須)
    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    df["_seq"] = df.groupby(key_cols).cumcount()

    return df.sort_values(key_cols + ["_seq"]).reset_index(drop=True)


def ascii_signature(s: str) -> str:
    """文字列からASCII部分だけを抽出したsignatureを返す

    ASCII(英数字・記号)は無傷で残るため、化けカラムの近似比較に使う
    """
    s = unicodedata.normalize("NFKC", s)  # 全角→半角
    return "".join(ALNUM_PATTERN.findall(s))


# ────────────────────────────────────────────────────────────────────
# 結果の入れ物

@dataclass(frozen=True)
class VerifyResult:
    only_left: pd.DataFrame  # golden にだけある行
    only_right: pd.DataFrame  # target にだけある行
    cell_diff: pd.DataFrame  # 両方にあるが値が違う行
    fuzzy_matched: pd.DataFrame  # 近似比較で一致扱いにしたセル

    @property
    def is_match(self) -> bool:
        return self.only_left.empty and self.only_right.empty and self.cell_diff.empty


# ────────────────────────────────────────────────────────────────────
# 突合本体

def verify(
        left: pd.DataFrame,  # golden 出力
        right: pd.DataFrame,  # 検証対象 (target 出力)
        key_cols: list[str],
        float_atol: float = FLOAT_ATOL,
) -> VerifyResult:
    """2つのDataFrameを突合し、行差分・セル差分を返す"""

    # 日付列は基準側(golden=left)だけで一度だけ決め、両方に同じリストを適用する
    # （左右で別々に判定すると非対称になり偽差分の原因になるため）
    date_cols = resolve_date_cols(left)
    manual = [c for c in DATE_COLS if c]
    print(f"🔖 date_cols: {date_cols} (manual={manual}, auto={[c for c in date_cols if c not in manual]})")

    left_n = normalize(left, key_cols, date_cols)
    right_n = normalize(right, key_cols, date_cols)

    merge_keys = key_cols + ["_seq"]

    # ────────────────────────────────────────────────────────────────────
    # 行単位の差分
    merged = left_n.merge(
        right_n, how="outer", on=merge_keys, indicator=True, suffixes=("_L", "_R")
    )
    only_left = merged[merged["_merge"] == "left_only"].drop(columns="_seq")
    only_right = merged[merged["_merge"] == "right_only"].drop(columns="_seq")

    # ────────────────────────────────────────────────────────────────────
    # セル単位の差分（両方に存在する行のみ）
    both_keys = merged.loc[merged["_merge"] == "both", merge_keys]
    left_both = left_n.merge(both_keys, on=merge_keys).set_index(merge_keys)
    right_both = right_n.merge(both_keys, on=merge_keys).set_index(merge_keys)
    right_both = right_both[left_both.columns]  # 列順をそろえる

    diffs: list[pd.DataFrame] = []
    fuzzy: list[pd.DataFrame] = []

    for col in left_both.columns:
        l, r = left_both[col], right_both[col]

        # --- step 1: 通常比較 ---
        if pd.api.types.is_numeric_dtype(l) and pd.api.types.is_numeric_dtype(r):
            mismatch = ~np.isclose(l, r, atol=float_atol, equal_nan=True)
        else:
            mismatch = (l.astype(str) != r.astype(str)).values

        # --- step 2: diff が出たセルだけASCIIで再判定 / 文字化けをスルーする
        if mismatch.any() and col in GARBLED_COLS:
            l_sig = l[mismatch].map(ascii_signature)
            r_sig = r[mismatch].map(ascii_signature)
            still_mismatch = (l_sig != r_sig).values

            # 救済されたセル（通常比較は不一致だが、signatureは一致したもの）を記録
            rescued_mask = mismatch.copy()
            rescued_mask[mismatch] = ~still_mismatch
            if rescued_mask.any():
                f = pd.DataFrame(
                    {"column": col, LEFT_KEY: l[rescued_mask], RIGHT_KEY: r[rescued_mask]}
                )
                fuzzy.append(f.reset_index())

            # 最終的な不一致
            final_mask = mismatch.copy()
            final_mask[mismatch] = still_mismatch
            mismatch = final_mask

        if mismatch.any():
            d = pd.DataFrame({"column": col, LEFT_KEY: l[mismatch], RIGHT_KEY: r[mismatch]})
            diffs.append(d.reset_index())

    cell_diff = pd.concat(diffs, ignore_index=True) if diffs else pd.DataFrame()
    fuzzy_matched = pd.concat(fuzzy, ignore_index=True) if fuzzy else pd.DataFrame()

    if not cell_diff.empty:
        cell_diff = cell_diff.drop(columns="_seq")
    if not fuzzy_matched.empty:
        fuzzy_matched = fuzzy_matched.drop(columns="_seq")

    return VerifyResult(
        only_left=only_left,
        only_right=only_right,
        cell_diff=cell_diff,
        fuzzy_matched=fuzzy_matched
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="golden出力とtarget出力の突合")
    parser.add_argument(
        "golden", nargs="?", default=DEFAULT_GOLDEN, type=Path,
        help=f"golden出力CSV （デフォルト: {DEFAULT_GOLDEN}）",
        )
    parser.add_argument(
        "target", nargs="?", default=DEFAULT_TARGET, type=Path,
        help="target出力CSV",
        )
    parser.add_argument(
        "--key", nargs="+", default=DEFAULT_KEYS, help="突合キー列",
        )
    args = parser.parse_args()

    left = read_csv(args.golden, "args.golden")
    right = read_csv(args.target, "args.target")

    result = verify(left, right, key_cols=args.key)

    print(f"-- RUN --\n\n🎈 {__name__}\n")
    mark_ok = "✅"
    mark_ng = "⚠️"
    n = 20

    mark = mark_ng
    if len(result.only_left) == 0:
        mark = mark_ok
    print(f"\n{mark} {LEFT_KEY} のみ: {len(result.only_left)}行")

    if not result.only_left.empty:
        my_columns = DEFAULT_KEYS
        subset = result.only_left[my_columns]
        print(f"  = top{n} =")
        print(subset.head(n).to_string())

    mark = mark_ng
    if len(result.only_right) == 0:
        mark = mark_ok
    print(f"\n{mark} {RIGHT_KEY} のみ: {len(result.only_right)}行")

    if not result.only_right.empty:
        my_columns = (
            DEFAULT_KEYS
            + [f"{c}_L" for c in EXTRA_COLS]
            + [f"{c}_R" for c in EXTRA_COLS]
        )
        subset = result.only_right[my_columns]
        print(f"  = top{n} =")
        print(subset.head(n).to_string())

    mark = mark_ng
    if len(result.fuzzy_matched) == 0:
        mark = mark_ok
    print(f"\n{mark} 近似値(文字化けスルー): {len(result.fuzzy_matched)}行")

    mark = mark_ng
    if len(result.cell_diff) == 0:
        mark = mark_ok
    print(f"\n{mark} 両方にあるが値が違う行: {len(result.cell_diff)}行")

    if not result.cell_diff.empty:
        print(f"  = top{n} =")
        print(result.cell_diff.head(n).to_string())

    if not result.fuzzy_matched.empty:
        unique_pairs = (
            result.fuzzy_matched[["column", LEFT_KEY, RIGHT_KEY]]
            .drop_duplicates()
            .sort_values(["column", LEFT_KEY])
            .reset_index(drop=True)
        )
        print(f"\n {mark_ng} 近似一致 / 文字化けだと思われるもの:")
        print(unique_pairs)


if __name__ == "__main__":
    main()
