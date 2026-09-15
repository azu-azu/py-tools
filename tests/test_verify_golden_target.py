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


if __name__ == "__main__":
    unittest.main()
