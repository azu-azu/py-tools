"""verify_golden_targetの回帰テスト。

実行方法::

    python -m unittest discover -s tests

pandasが要る以外の依存はない。
CSVはすべてテスト内で生成した合成データで、実データは使わない。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


# scriptsはパッケージではないので、パス指定で読み込む
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_golden_target.py"
)

_spec = importlib.util.spec_from_file_location(
    "verify_golden_target",
    _MODULE_PATH,
)
vgt = importlib.util.module_from_spec(_spec)
sys.modules["verify_golden_target"] = vgt
_spec.loader.exec_module(vgt)


def _silently(func, *args, **kwargs):
    """突合の進捗printを飲み込んで実行する。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _captured(func, *args, **kwargs) -> str:
    """printされた内容を文字列として受け取る。"""
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)

    return buffer.getvalue()


class NormalizeMixedDtypeTest(unittest.TestCase):
    """_normalizeが混在dtype列を壊さないことを確認する。

    read_csvはチャンク単位で型を推論するため、同じobject列の中に
    int と str が混ざることがある。

    その列に.str.strip()を列ごと掛けると、str以外のセルがNaNになり、
    直後のfillna("")で空文字へ化けて値が消える。
    """

    def test_mixed_str_and_int_column_keeps_int(self) -> None:
        df = pd.DataFrame(
            {"C": pd.Series([" 4 ", 4, None], dtype=object)}
        )

        normalized = vgt._normalize(df, [], sort_rows=False)

        self.assertEqual(
            list(normalized["C"]),
            ["4", 4, ""],
        )

    def test_all_int_object_column_does_not_raise(self) -> None:
        # 全セルがintのobject列は.strアクセサ自体が
        # AttributeErrorで落ちていた
        df = pd.DataFrame(
            {"C": pd.Series([4, 5], dtype=object)}
        )

        normalized = vgt._normalize(df, [], sort_rows=False)

        self.assertEqual(list(normalized["C"]), [4, 5])

    def test_pure_string_column_is_still_stripped(self) -> None:
        # ベクトル化された高速パス側の確認
        df = pd.DataFrame(
            {"C": pd.Series([" a ", "b", np.nan], dtype=object)}
        )

        normalized = vgt._normalize(df, [], sort_rows=False)

        self.assertEqual(
            list(normalized["C"]),
            ["a", "b", ""],
        )


class VerifyMixedDtypeTest(unittest.TestCase):
    """左右で同じ値が別のdtypeで読まれても一致すること。

    goldenがint 4、targetがstr "4"のように型が割れるのは、
    ファイルごとに型推論が走る以上避けられない。

    値が同じなら一致と判定されなければならない。
    """

    def test_int_column_matches_str_column(self) -> None:
        left = pd.DataFrame(
            {
                "ID": [1, 2],
                "Floor": pd.Series([4, 4]),  # int64
            }
        )
        right = pd.DataFrame(
            {
                "ID": [1, 2],
                "Floor": pd.Series(["4", "4"], dtype=object),
            }
        )

        result = _silently(
            vgt._verify,
            left,
            right,
            key_cols=["ID"],
        )

        self.assertTrue(result.is_match)

    def test_mixed_column_matches_str_column(self) -> None:
        # 実際に壊れていた形。左はintとstrが混在したobject列
        left = pd.DataFrame(
            {
                "ID": [1, 2],
                "Floor": pd.Series([" 4 ", 4], dtype=object),
            }
        )
        right = pd.DataFrame(
            {
                "ID": [1, 2],
                "Floor": pd.Series(["4", "4"], dtype=object),
            }
        )

        result = _silently(
            vgt._verify,
            left,
            right,
            key_cols=["ID"],
        )

        self.assertTrue(result.is_match)

    def test_real_difference_is_still_detected(self) -> None:
        # 何でも一致してしまう修正になっていないことの担保
        left = pd.DataFrame(
            {
                "ID": [1, 2],
                "Floor": pd.Series([4, 4], dtype=object),
            }
        )
        right = pd.DataFrame(
            {
                "ID": [1, 2],
                "Floor": pd.Series(["4", "5"], dtype=object),
            }
        )

        result = _silently(
            vgt._verify,
            left,
            right,
            key_cols=["ID"],
        )

        self.assertFalse(result.is_match)
        self.assertEqual(len(result.cell_diff), 1)


class ReadCsvChunkBoundaryTest(unittest.TestCase):
    """チャンク境界をまたいでも列の型が割れないことを確認する。

    low_memory=Trueだと、pandasのCパーサは262,144行ごとに
    型を推論するため、前半がint・後半がstrのobject列ができる。

    どこで割れるかは行数と行順まかせなので、
    golden側とtarget側で別々の行が壊れて偽差分になっていた。
    """

    CHUNK_ROWS = 262_144

    def test_column_dtype_is_uniform_across_chunk_boundary(self) -> None:
        rows = self.CHUNK_ROWS + 100

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.csv"

            with path.open("w", encoding="cp932") as f:
                f.write("ID,Floor\n")

                for index in range(rows):
                    # 文字列は最終チャンクにだけ現れる
                    floor = "4" if index < rows - 50 else "B1F"
                    f.write(f"{index},{floor}\n")

            df = _silently(vgt._read_csv, path, "synthetic")

        types = {type(value).__name__ for value in df["Floor"]}

        self.assertEqual(types, {"str"})

        normalized = vgt._normalize(df, [], sort_rows=False)

        self.assertEqual((normalized["Floor"] == "").sum(), 0)


class CellDiffByColumnTest(unittest.TestCase):
    """セル差分の列ごとの件数表を確認する。

    明細はtop20までしか出ないため、差分がどの列に偏っているかは
    件数表でしか分からない。並びが実行のたびに変わると
    前回の出力と見比べられないので、並びまで含めて固定する。
    """

    def _result(self):
        left = pd.DataFrame(
            {
                "ID": ["A", "B", "C"],
                "name": ["p", "q", "r"],
                "amount": [1, 2, 3],
                "memo": ["x", "y", "z"],
            }
        )
        right = left.copy()

        # amount 2件、name 1件、memo 1件
        right.loc[0, "amount"] = 11
        right.loc[1, "amount"] = 22
        right.loc[0, "name"] = "P"
        right.loc[2, "memo"] = "Z"

        return _silently(
            vgt._verify,
            left,
            right,
            key_cols=["ID"],
        )

    def test_counts_sum_matches_cell_diff(self) -> None:
        result = self._result()
        counts = result.cell_diff_by_column

        self.assertEqual(
            list(counts.columns),
            ["column", "count"],
        )
        self.assertEqual(
            int(counts["count"].sum()),
            len(result.cell_diff),
        )

    def test_ties_keep_original_column_order(self) -> None:
        # nameとmemoは同数。goldenの列順がnameを先に置いているので、
        # 件数表でもnameが先に来る
        counts = self._result().cell_diff_by_column

        self.assertEqual(
            list(counts["column"]),
            ["amount", "name", "memo"],
        )
        self.assertEqual(
            list(counts["count"]),
            [2, 1, 1],
        )

    def test_empty_when_no_diff(self) -> None:
        left = pd.DataFrame({"ID": ["A"], "x": [1]})

        result = _silently(
            vgt._verify,
            left,
            left.copy(),
            key_cols=["ID"],
        )
        counts = result.cell_diff_by_column

        self.assertTrue(counts.empty)
        self.assertEqual(
            list(counts.columns),
            ["column", "count"],
        )


class CellDiffRowCountTest(unittest.TestCase):
    """cell_diffのセル数と行数を取り違えないことを確認する。

    cell_diffは1行1セルなので、len(cell_diff)は行数にならない。
    同一キーの重複行は_seqで対応づけており、その_seqは
    cell_diffに残らないため、行数は_verifyが数えて持っている。
    """

    def test_multiple_columns_in_one_row_count_as_one_row(self) -> None:
        left = pd.DataFrame(
            {
                "ID": ["A", "B"],
                "x": [1, 2],
                "y": ["p", "q"],
                "z": [7, 8],
            }
        )
        right = left.copy()
        right.loc[0, ["x", "y", "z"]] = [9, "Z", 99]

        result = _silently(
            vgt._verify,
            left,
            right,
            key_cols=["ID"],
        )

        self.assertEqual(len(result.cell_diff), 3)
        self.assertEqual(result.cell_diff_rows, 1)

    def test_duplicate_keys_are_counted_separately(self) -> None:
        # 同一キーが3行。_seqを落としてから数えると1行に潰れる
        left = pd.DataFrame(
            {
                "ID": ["A", "A", "A"],
                "x": [1, 2, 3],
            }
        )
        right = pd.DataFrame(
            {
                "ID": ["A", "A", "A"],
                "x": [11, 22, 33],
            }
        )

        result = _silently(
            vgt._verify,
            left,
            right,
            key_cols=["ID"],
        )

        self.assertEqual(len(result.cell_diff), 3)
        self.assertEqual(result.cell_diff_rows, 3)

    def test_no_diff_has_zero_rows(self) -> None:
        left = pd.DataFrame({"ID": ["A"], "x": [1]})

        result = _silently(
            vgt._verify,
            left,
            left.copy(),
            key_cols=["ID"],
        )

        self.assertEqual(result.cell_diff_rows, 0)


class PrintResultCellDiffTest(unittest.TestCase):
    """セル差分の表示内容を確認する。"""

    def _printed(self, left, right) -> str:
        result = _silently(
            vgt._verify,
            left,
            right,
            key_cols=["ID"],
        )

        return _captured(
            vgt._print_result,
            result,
            key_cols=["ID"],
            max_rows=20,
        )

    def test_heading_shows_cells_and_rows(self) -> None:
        left = pd.DataFrame(
            {
                "ID": ["A", "B"],
                "x": [1, 2],
                "y": ["p", "q"],
            }
        )
        right = left.copy()
        right.loc[0, ["x", "y"]] = [9, "Z"]

        printed = self._printed(left, right)

        self.assertIn(
            "両方にあるが値が違うセル: 2件 (1行)",
            printed,
        )
        self.assertIn("= 列ごとの差分件数 =", printed)

    def test_no_diff_prints_zero_without_counts(self) -> None:
        left = pd.DataFrame({"ID": ["A"], "x": [1]})

        printed = self._printed(left, left.copy())

        self.assertIn(
            "両方にあるが値が違うセル: 0件",
            printed,
        )
        self.assertNotIn("= 列ごとの差分件数 =", printed)


def _residual_pair(left_rows, right_rows, columns):
    """同一キーの残差を作るための小さなDataFrameの組。"""
    return (
        pd.DataFrame(left_rows, columns=columns),
        pd.DataFrame(right_rows, columns=columns),
    )


class PairingAmbiguityTest(unittest.TestCase):
    """Stage 2の行対応が一意かどうかの診断を確認する。

    Stage 1で全列一致行を吸収した後、同じキーに複数行が残ると、
    どの行同士を突き合わせるかは一意に決まらない。

    差分そのものは本物である（行として一致する相手は残っていない）。
    曖昧なのは、その差分をどの列に何件として数えるかのほう。
    """

    def _ambiguity(self, left, right, keys=("ID",)):
        result = _silently(
            vgt._verify,
            left,
            right,
            key_cols=list(keys),
        )

        return result, result.pairing_ambiguity

    def test_unique_key_has_no_ambiguity(self) -> None:
        left = pd.DataFrame({"ID": ["A", "B"], "x": [1, 2]})
        right = pd.DataFrame({"ID": ["A", "B"], "x": [1, 9]})

        result, ambiguity = self._ambiguity(left, right)

        self.assertEqual(len(result.cell_diff), 1)
        self.assertEqual(ambiguity.ambiguous_cells, 0)
        self.assertEqual(ambiguity.ambiguous_keys, 0)
        self.assertEqual(ambiguity.duplicate_key_groups, 0)
        self.assertFalse(ambiguity.has_ambiguity)

    def test_reordered_rows_leave_no_residual(self) -> None:
        # Stage 1が順序に依存せず吸収するので、残差自体が生まれない
        left = pd.DataFrame(
            {"ID": ["K"] * 3, "x": [1, 2, 3]}
        )
        right = left.iloc[[2, 0, 1]].reset_index(drop=True)

        result, ambiguity = self._ambiguity(left, right)

        self.assertTrue(result.is_match)
        self.assertEqual(ambiguity.duplicate_key_groups, 0)
        self.assertFalse(ambiguity.has_ambiguity)

    def test_one_to_one_residual_is_unique_pairing(self) -> None:
        # 同一キーが複数行でも、残差が左右1行ずつなら対応は決まる
        left = pd.DataFrame(
            {"ID": ["K", "K"], "x": [1, 2]}
        )
        right = pd.DataFrame(
            {"ID": ["K", "K"], "x": [1, 9]}
        )

        result, ambiguity = self._ambiguity(left, right)

        self.assertEqual(len(result.cell_diff), 1)
        self.assertFalse(ambiguity.has_ambiguity)
        self.assertEqual(ambiguity.duplicate_key_groups, 0)

    def test_two_to_two_residual_is_ambiguous(self) -> None:
        left, right = _residual_pair(
            [["K", 1, "P"], ["K", 2, "Q"]],
            [["K", 1, "Q"], ["K", 2, "P"]],
            ["ID", "a", "b"],
        )

        result, ambiguity = self._ambiguity(left, right)

        # 差分は従来どおり保持する
        self.assertEqual(len(result.cell_diff), 2)
        self.assertEqual(ambiguity.ambiguous_cells, 2)
        self.assertEqual(ambiguity.ambiguous_rows, 2)
        self.assertEqual(ambiguity.ambiguous_keys, 1)
        self.assertEqual(ambiguity.duplicate_key_groups, 1)
        self.assertTrue(ambiguity.has_ambiguity)
        self.assertTrue(ambiguity.has_ambiguous_cell_diff)

    def test_two_to_one_residual_is_ambiguous(self) -> None:
        # ペアが1組できて1行あぶれる。どの行があぶれるかも対応づけ次第
        left, right = _residual_pair(
            [["K", 1, "X"], ["K", 2, "Y"]],
            [["K", 1, "Z"]],
            ["ID", "a", "b"],
        )

        result, ambiguity = self._ambiguity(left, right)

        self.assertEqual(len(result.only_left), 1)
        self.assertTrue(ambiguity.has_ambiguity)
        self.assertEqual(ambiguity.ambiguous_keys, 1)
        self.assertEqual(ambiguity.max_left_rows_per_key, 2)
        self.assertEqual(ambiguity.max_right_rows_per_key, 1)

    def test_one_sided_residual_is_not_ambiguous(self) -> None:
        # 片側が0行ならペアが作られない。
        # 2行ともonly_leftへ行くだけで、対応づけの選択は起きていない
        left = pd.DataFrame(
            {"ID": ["K", "K", "J"], "x": [1, 2, 3]}
        )
        right = pd.DataFrame({"ID": ["J"], "x": [3]})

        result, ambiguity = self._ambiguity(left, right)

        self.assertEqual(len(result.only_left), 2)
        self.assertEqual(ambiguity.ambiguous_keys, 0)
        self.assertFalse(ambiguity.has_ambiguity)

        # キーが行を一意に識別できていないこと自体は伝える
        self.assertEqual(ambiguity.duplicate_key_groups, 1)
        self.assertTrue(ambiguity.has_duplicate_keys)

    def test_column_names_change_blame_but_not_diagnosis(self) -> None:
        """列名を変えるとどの列が差分になるかが変わる。

        Stage 2は共通列名のアルファベット順で残差を並べるため、
        中身が同じでも列名次第でペアの組み方が変わる。

        診断はその並びに依存してはならない。
        """
        blamed = []
        diagnoses = []

        for first, second in (("a", "z"), ("m", "b")):
            left, right = _residual_pair(
                [["K", 1, "P"], ["K", 2, "Q"]],
                [["K", 1, "Q"], ["K", 2, "P"]],
                ["ID", first, second],
            )

            result, ambiguity = self._ambiguity(left, right)

            blamed.append(set(result.cell_diff["column"]))
            diagnoses.append(ambiguity)

        # 列名を変えただけで、差分として名指しされる列が入れ替わる
        self.assertEqual(blamed[0], {"z"})
        self.assertEqual(blamed[1], {"m"})

        # 診断は変わらない
        self.assertEqual(diagnoses[0], diagnoses[1])
        self.assertTrue(diagnoses[0].has_ambiguity)

    def test_multiset_match_keeps_the_difference(self) -> None:
        # 列ごとの値集合は左右で一致するが、行としては別物。
        # 「入れ替わっただけ」に見えても差分は消さない
        left, right = _residual_pair(
            [["K", 10, "A"], ["K", 20, "B"]],
            [["K", 20, "A"], ["K", 10, "B"]],
            ["ID", "amount", "memo"],
        )

        result, ambiguity = self._ambiguity(left, right)

        for col in ("amount", "memo"):
            self.assertEqual(
                sorted(left[col]),
                sorted(right[col]),
            )

        self.assertFalse(result.is_match)
        self.assertGreater(len(result.cell_diff), 0)
        self.assertTrue(ambiguity.has_ambiguity)

    def test_ambiguous_cell_diff_implies_not_match(self) -> None:
        # 曖昧なキーからセル差分が出ているなら、一致ではありえない
        left, right = _residual_pair(
            [["K", 1, "P"], ["K", 2, "Q"]],
            [["K", 1, "Q"], ["K", 2, "P"]],
            ["ID", "a", "b"],
        )

        result, ambiguity = self._ambiguity(left, right)

        self.assertGreater(ambiguity.ambiguous_cells, 0)
        self.assertTrue(ambiguity.has_ambiguous_cell_diff)
        self.assertFalse(result.is_match)

    def test_fuzzy_only_ambiguous_pairing_can_still_match(self) -> None:
        """曖昧さと「一致」は両立する。

        同一キーに2行ずつ残り、対応づけは一意に決まらないが、
        差分が全て文字化け吸収に救われるとcell_diffは0件になる。

        このときis_matchはTrueだが、その一致は
        どの行とどの行を突き合わせたか次第で成立している。
        セル差分の有無でhas_ambiguityを決めると、ここを取り落とす。
        """
        left = pd.DataFrame(
            {"ID": ["K", "K"], "C": ["A1あ", "A2あ"]}
        )
        right = pd.DataFrame(
            {"ID": ["K", "K"], "C": ["A1い", "A2い"]}
        )

        with mock.patch.object(vgt, "GARBLED_COLS", ["C"]):
            result, ambiguity = self._ambiguity(left, right)

        self.assertTrue(result.is_match)
        self.assertTrue(result.cell_diff.empty)
        self.assertEqual(len(result.fuzzy_matched), 2)

        self.assertEqual(ambiguity.ambiguous_keys, 1)
        self.assertEqual(ambiguity.ambiguous_cells, 0)
        self.assertTrue(ambiguity.has_ambiguity)
        self.assertFalse(ambiguity.has_ambiguous_cell_diff)

    def test_keyless_mode_has_no_diagnosis(self) -> None:
        # キーなしモードは行を対応づけないので診断対象外
        left = pd.DataFrame({"ID": ["A", "B"], "x": [1, 2]})
        right = pd.DataFrame({"ID": ["A", "B"], "x": [1, 9]})

        result, ambiguity = self._ambiguity(left, right, keys=())

        self.assertIsNone(ambiguity)
        self.assertIsNone(result.pairing_ambiguity)

    def test_garbled_absorption_is_untouched(self) -> None:
        # 文字化け吸収は従来どおり動き、診断も0のまま
        left = pd.DataFrame({"ID": ["A"], "C": ["A1あ"]})
        right = pd.DataFrame({"ID": ["A"], "C": ["A1い"]})

        with mock.patch.object(vgt, "GARBLED_COLS", ["C"]):
            result, ambiguity = self._ambiguity(left, right)

        self.assertTrue(result.cell_diff.empty)
        self.assertEqual(len(result.fuzzy_matched), 1)

        # キーが一意なので、そもそも曖昧さは生まれない
        self.assertFalse(ambiguity.has_ambiguity)
        self.assertFalse(ambiguity.has_ambiguous_cell_diff)


class PrintPairingNotesTest(unittest.TestCase):
    """曖昧さの注記が、必要なときだけ出ることを確認する。"""

    def _printed(self, left, right, keys=("ID",)) -> str:
        result = _silently(
            vgt._verify,
            left,
            right,
            key_cols=list(keys),
        )

        return _captured(
            vgt._print_result,
            result,
            key_cols=list(keys),
            max_rows=20,
        )

    def test_unique_key_prints_no_notes(self) -> None:
        left = pd.DataFrame({"ID": ["A", "B"], "x": [1, 2]})
        right = pd.DataFrame({"ID": ["A", "B"], "x": [1, 9]})

        printed = self._printed(left, right)

        self.assertNotIn("行対応が一意でない", printed)
        self.assertNotIn("同じキーに複数行が残る", printed)

    def test_ambiguous_key_prints_both_notes(self) -> None:
        left, right = _residual_pair(
            [["K", 1, "P"], ["K", 2, "Q"]],
            [["K", 1, "Q"], ["K", 2, "P"]],
            ["ID", "a", "b"],
        )

        printed = self._printed(left, right)

        self.assertIn("2件 (2行) は行対応が一意でない", printed)
        self.assertIn("同じキーに複数行が残る", printed)

    def test_ambiguity_is_never_silent(self) -> None:
        """一致と出ても、曖昧さがあるなら注記は消えない。

        ambiguousなキーは必ずduplicate_key_groupsにも数えられるため、
        セル差分が0件でもキー重複の注記が残る。
        「✅なので読まなくていい」にならないことの担保。
        """
        left = pd.DataFrame(
            {"ID": ["K", "K"], "C": ["A1あ", "A2あ"]}
        )
        right = pd.DataFrame(
            {"ID": ["K", "K"], "C": ["A1い", "A2い"]}
        )

        with mock.patch.object(vgt, "GARBLED_COLS", ["C"]):
            printed = self._printed(left, right)

        self.assertIn("値が違うセル: 0件", printed)
        self.assertIn("同じキーに複数行が残る", printed)

    def test_duplicate_key_note_without_cell_diff(self) -> None:
        # セル差分が0でも、キーが行を識別できていないことは伝える
        left = pd.DataFrame(
            {"ID": ["K", "K", "J"], "x": [1, 2, 3]}
        )
        right = pd.DataFrame({"ID": ["J"], "x": [3]})

        printed = self._printed(left, right)

        self.assertIn("値が違うセル: 0件", printed)
        self.assertIn("同じキーに複数行が残る", printed)
        self.assertNotIn("行対応が一意でない", printed)


if __name__ == "__main__":
    unittest.main()
