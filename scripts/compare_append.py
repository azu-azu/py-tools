"""golden と target（append 済み）の突合スクリプト。

Find Replace（Find Any Part of Field）系の付与結果を golden と突き合わせるための道具。
verify_golden_target.py と違い、比較したい append 列「以外」の全列で行のペアを決め、
append 列は比較専用にする。これにより:

  - 並び順が golden と target で違っても差分が出ない（順序非依存）
  - 業務キーが重複していても、append 値の差でペアが崩れて波及することがない

target 側は golden と同じ列構成（= 元データ列 + APPEND_FIELDS）にしておくこと。
_target_row_id / _source_row_id / 検索値エコー等の内部列は事前に drop しておく。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ────────────────────────────────────────────────────────────────────
# 設定　都度変更する

# 突合したい付与列（この列「以外」が行のペアを決めるキーになる）
APPEND_FIELDS: list[str] = [""]

DATA_DIR = Path(__file__).resolve().parents[1]  # py-tools プロジェクト直下
DEFAULT_GOLDEN = DATA_DIR / "golden.csv"
DEFAULT_TARGET = DATA_DIR / "output" / "debug" / "_test.csv"

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
# 正規化

def normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """比較に使う列を文字列化・前後空白除去・NaN→"" に統一する。"""
    out = df[columns].copy()
    for col in columns:
        s = out[col]
        out[col] = s.where(s.notna(), "").astype(str).str.strip()
    return out


# ────────────────────────────────────────────────────────────────────
# 結果の入れ物

@dataclass(frozen=True)
class CompareResult:
    only_golden: pd.DataFrame   # golden にしかない行（元データ列が target と食い違う）
    only_target: pd.DataFrame   # target にしかない行
    append_diff: pd.DataFrame   # 行は対応するが append 値が違うもの
    pass_cols: list[str]        # ペアを決めるのに使った列

    @property
    def is_match(self) -> bool:
        return self.only_golden.empty and self.only_target.empty and self.append_diff.empty


# ────────────────────────────────────────────────────────────────────
# 突合本体

def compare_append(
    golden: pd.DataFrame,
    target: pd.DataFrame,
    append_fields: list[str],
) -> CompareResult:
    """append 以外の全列で行をペアにし、append 列だけを比較する。"""

    # ペアを決めるキー = append 以外の全列（＝安定した行の身元）
    pass_cols = [c for c in golden.columns if c not in append_fields]

    missing_in_golden = [c for c in append_fields if c not in golden.columns]
    missing_in_target = [
        c for c in [*pass_cols, *append_fields] if c not in target.columns
    ]
    if missing_in_golden:
        raise KeyError(f"golden に列がありません: {missing_in_golden}")
    if missing_in_target:
        raise KeyError(f"target に列がありません: {missing_in_target}")

    need = [*pass_cols, *append_fields]
    g = normalize(golden, need)
    t = normalize(target, need)

    # pass_cols だけで安定ソート → 完全重複行に連番 _seq を振る。
    # append 列はソートに含めないので、値の差でペアがズレることがない。
    def keyed(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(pass_cols, kind="stable").reset_index(drop=True)
        df["_seq"] = df.groupby(pass_cols).cumcount()
        return df

    g = keyed(g)
    t = keyed(t)

    merge_keys = [*pass_cols, "_seq"]
    merged = g.merge(
        t, how="outer", on=merge_keys, indicator=True, suffixes=("_g", "_t")
    )

    only_golden = merged[merged["_merge"] == "left_only"]
    only_target = merged[merged["_merge"] == "right_only"]

    both = merged[merged["_merge"] == "both"]
    diff_mask = pd.Series(False, index=both.index)
    for col in append_fields:
        diff_mask |= both[f"{col}_g"] != both[f"{col}_t"]
    append_diff = both[diff_mask]

    display_cols = (
        pass_cols
        + [f"{c}_g" for c in append_fields]
        + [f"{c}_t" for c in append_fields]
    )

    return CompareResult(
        only_golden=only_golden[pass_cols],
        only_target=only_target[pass_cols],
        append_diff=append_diff[display_cols],
        pass_cols=pass_cols,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="golden と append 済み target の突合")
    parser.add_argument(
        "golden", nargs="?", default=DEFAULT_GOLDEN, type=Path,
        help=f"golden CSV （デフォルト: {DEFAULT_GOLDEN}）",
    )
    parser.add_argument(
        "target", nargs="?", default=DEFAULT_TARGET, type=Path,
        help=f"append 済み target CSV （デフォルト: {DEFAULT_TARGET}）",
    )
    parser.add_argument(
        "--append", nargs="+", default=[c for c in APPEND_FIELDS if c],
        help="突合する付与列（デフォルト: APPEND_FIELDS）",
    )
    args = parser.parse_args()

    append_fields = args.append
    if not append_fields:
        parser.error("append 列が未設定です。APPEND_FIELDS か --append で指定してください。")

    golden = read_csv(args.golden, "golden")
    target = read_csv(args.target, "target")

    result = compare_append(golden, target, append_fields)

    print(f"\n-- RUN --\n\n🎈 compare_append\n")
    print(f"append 列   : {append_fields}")
    print(f"ペアキー列  : {result.pass_cols}")
    print(f"golden 行数 : {len(golden):,}")
    print(f"target 行数 : {len(target):,}")

    n = 30
    mark_ok, mark_ng = "✅", "⚠️"

    mark = mark_ok if result.only_golden.empty else mark_ng
    print(f"\n{mark} golden のみ行: {len(result.only_golden)}行")
    if not result.only_golden.empty:
        print(result.only_golden.head(n).to_string(index=False))

    mark = mark_ok if result.only_target.empty else mark_ng
    print(f"\n{mark} target のみ行: {len(result.only_target)}行")
    if not result.only_target.empty:
        print(result.only_target.head(n).to_string(index=False))

    mark = mark_ok if result.append_diff.empty else mark_ng
    print(f"\n{mark} append 値が違う行: {len(result.append_diff)}行")
    if not result.append_diff.empty:
        print(result.append_diff.head(n).to_string(index=False))

    print(f"\n{'✅ 完全一致' if result.is_match else '⚠️ 差分あり'}")


if __name__ == "__main__":
    main()
