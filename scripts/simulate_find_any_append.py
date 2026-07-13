"""Alteryx「Find Replace」(Find Any Part of Field) を pandas で再現する検証スクリプト。

targets_df の find_field に、source_df の search_field の値が
部分文字列として含まれていれば、その source 行の append_fields を付与する。
1つの target が複数の source 行にマッチした場合、どの行の値を採用するかは
replace_multiple_found で決まる（Alteryx の ReplaceMultipleFound 設定に対応）。

cross join + apply(axis=1) は O(n×m) の Python ループになって重いので、
source（たいてい小さい方）を m 回ループし、各 needle について
targets 側を str.contains でベクトル判定する（ベクトル化パス m 回で済む）。
"""

from __future__ import annotations

import time

import pandas as pd

# 元データ・ルックアップ表の行を追跡するための内部 ID 列
TARGET_ROW_ID = "_target_row_id"
SOURCE_ROW_ID = "_source_row_id"


def simulate_find_any_append(
    targets_df: pd.DataFrame,   # 残したい元データ
    source_df: pd.DataFrame,    # ルックアップ表（探す値と追加列を持つ）
    *,
    find_field: str,
    search_field: str,
    append_fields: list[str],
    case_sensitive: bool = True,  # Alteryx の NoCase=False（大小を区別）に対応
    replace_multiple_found: bool = True,  # Alteryx の ReplaceMultipleFound。True=last match、False=first match
    verbose: bool = True,
) -> pd.DataFrame:
    """find_field に search_field 値を部分一致で探し、マッチした append_fields を付与する。

    Find Replace は join ではないので、複数マッチしても出力は 1 target = 1 行。
    複数の source 行にマッチしたときにどの行の値を採用するかは
    replace_multiple_found で決まる（Alteryx の ReplaceMultipleFound 設定に対応）:
    True なら source 順で最後にマッチした行、False なら最初にマッチした行。
    """

    start = time.perf_counter()

    # ── 入力チェック ──────────────────────────────────────────────
    if find_field not in targets_df.columns:
        raise KeyError(f"targets_df に列がありません: {find_field}")

    required_source_columns = [search_field, *append_fields]
    missing_source_columns = [
        column
        for column in required_source_columns
        if column not in source_df.columns
    ]
    if missing_source_columns:
        raise KeyError(
            f"source_df に列がありません: {missing_source_columns}"
        )

    # 付与する列（search_field を含む）が targets 側に既にあると結果が壊れるので弾く。
    # find_field == search_field のケースもここで検出できる。
    new_columns = [search_field, *append_fields]
    overlap = [column for column in new_columns if column in targets_df.columns]
    if overlap:
        raise ValueError(
            f"付与する列が targets_df 側に既に存在しています: {overlap}"
        )

    # ── 準備 ─────────────────────────────────────────────────────
    targets = targets_df.reset_index(drop=True)
    targets.insert(0, TARGET_ROW_ID, range(len(targets)))

    source = source_df[required_source_columns].reset_index(drop=True)

    # find_field を文字列化した haystack。NaN は NaN のまま残す
    # （astype(str) だけだと NaN が "nan" になり誤マッチするため where で戻す）。
    raw_find = targets[find_field]
    haystack = raw_find.astype(str).where(raw_find.notna(), other=pd.NA)
    haystack_cmp = haystack if case_sensitive else haystack.str.lower()

    # ── マッチ結果の入れ物 ─────────────────────────────────────────
    winning_source_id = pd.Series(pd.NA, index=targets.index, dtype="object")
    matched_needle = pd.Series(pd.NA, index=targets.index, dtype="object")
    appended = {
        field: pd.Series(pd.NA, index=targets.index, dtype="object")
        for field in append_fields
    }
    match_count = pd.Series(0, index=targets.index, dtype="int64")  # 確認用: 何行の source にマッチしたか
    unmatched = pd.Series(True, index=targets.index)  # first match モードでまだ確定していない行

    # source を並び順にループ。itertuples の 0 番目が search_field、以降が append_fields。
    append_positions = range(1, len(required_source_columns))
    for source_id, values in enumerate(source.itertuples(index=False, name=None)):
        needle = values[0]
        if pd.isna(needle):
            continue
        needle = str(needle)
        if not needle:
            continue

        needle_cmp = needle if case_sensitive else needle.lower()
        contains = haystack_cmp.str.contains(needle_cmp, regex=False, na=False)
        if not contains.any():
            continue

        # 診断用は「何行の source にマッチしたか」なので確定済みも含めて数える
        match_count += contains.astype("int64")

        # 値を埋める対象。ReplaceMultipleFound=True(last match) はマッチのたびに上書きし、
        # source 順で最後にマッチした行が残る。False(first match) は未確定の行だけ埋める。
        fill = contains if replace_multiple_found else (contains & unmatched)
        if fill.any():
            winning_source_id[fill] = source_id
            matched_needle[fill] = needle
            for position, field in zip(append_positions, append_fields):
                appended[field][fill] = values[position]
        if not replace_multiple_found:
            unmatched &= ~contains

    # ── 結果の組み立て（入力順のまま。matched/unmatched に分割しない）──
    result = targets.copy()
    result[SOURCE_ROW_ID] = winning_source_id.astype("Int64")
    result[search_field] = matched_needle
    for field in append_fields:
        result[field] = appended[field]

    result_columns = [
        TARGET_ROW_ID,
        *targets_df.columns,
        SOURCE_ROW_ID,
        search_field,
        *append_fields,
    ]
    result = result[result_columns]

    if verbose:
        _print_summary(
            start=start,
            targets_df=targets_df,
            result=result,
            match_count=match_count,
            matched_needle=matched_needle,
            search_field=search_field,
        )

    return result


def _print_summary(
    *,
    start: float,
    targets_df: pd.DataFrame,
    result: pd.DataFrame,
    match_count: pd.Series,
    matched_needle: pd.Series,
    search_field: str,
) -> None:
    """処理時間・行数・複数マッチ（曖昧マッチ）の確認用サマリを出す。"""

    elapsed = time.perf_counter() - start
    matched_rows = int((match_count > 0).sum())

    print(f"\n 🐒 simulate_find_any_append: {elapsed:.3f} 秒")
    print(f"rows before   : {len(targets_df):,}")
    print(f"rows after    : {len(result):,}")
    print(f"matched rows  : {matched_rows:,}")

    # 1 target が複数 source 行にマッチした（＝採用値が source 順に依存する）行を可視化
    ambiguous = pd.DataFrame(
        {
            TARGET_ROW_ID: result[TARGET_ROW_ID],
            "matched_lookup_rows": match_count.to_numpy(),
            f"chosen_{search_field}": matched_needle.to_numpy(),
        }
    )
    ambiguous = ambiguous[ambiguous["matched_lookup_rows"] > 1].sort_values(
        "matched_lookup_rows", ascending=False
    )
    print(f"ambiguous rows: {len(ambiguous):,}（複数 source にマッチ）")
    if not ambiguous.empty:
        print(ambiguous.head(10).to_string(index=False))
    print()
