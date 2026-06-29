import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

SHEETS = {
    "L": {
        "sheet_name": "L_left_only",
        "table_name": "LLeftOnlyTable",
        "description": "CSVにだけ存在",
        "merge_value": "left_only",
    },
    "J": {
        "sheet_name": "J_joined",
        "table_name": "JJoinedTable",
        "description": "CSVとExcelマスタの両方に存在",
        "merge_value": "both",
    },
    "R": {
        "sheet_name": "R_right_only",
        "table_name": "RRightOnlyTable",
        "description": "Excelマスタにだけ存在",
        "merge_value": "right_only",
    },
}


def make_unique_columns(data: pd.DataFrame) -> pd.DataFrame:
    table_df = data.copy()
    seen: dict[str, int] = {}
    new_columns = []

    for col in table_df.columns:
        base = str(col).strip() or "Column"
        key = base.lower()
        count = seen.get(key, 0)

        new_columns.append(base if count == 0 else f"{base}_{count}")
        seen[key] = count + 1

    table_df.columns = pd.Index(new_columns)
    return table_df


def write_sheet_as_table(
    writer: pd.ExcelWriter,
    data: pd.DataFrame,
    sheet_name: str,
    table_name: str,
) -> None:
    table_df = make_unique_columns(data)
    table_df.to_excel(writer, sheet_name=sheet_name, index=False)

    worksheet = writer.sheets[sheet_name]
    worksheet.hide_gridlines(2)

    rows, cols = table_df.shape
    if cols == 0:
        return

    columns = [{"header": str(col)} for col in table_df.columns]
    worksheet.add_table(
        0,
        0,
        rows,
        cols - 1,
        {
            "name": table_name,
            "columns": columns,
            "style": "Table Style Light 1",
            "banded_rows": False,
            "banded_columns": False,
        },
    )

    for col_idx, col_name in enumerate(table_df.columns):
        values = table_df.iloc[:, col_idx].dropna().head(1000)
        max_len = max([len(str(col_name))] + [len(str(v)) for v in values])
        worksheet.set_column(col_idx, col_idx, min(max_len + 2, 60))


def open_file(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


app = typer.Typer()


@app.command()
def main(
    csv_path: Annotated[Path, typer.Option("--csv", help="入力 CSV ファイル")],
    master_path: Annotated[Path, typer.Option("--master", help="マスタ Excel ファイル")],
    csv_keys: Annotated[list[str], typer.Option("--csv-key", help="CSV 側の join key（複数指定可）")],
    master_keys: Annotated[list[str], typer.Option("--master-key", help="マスタ側の join key（複数指定可）")],
    master_sheet: Annotated[str, typer.Option("--sheet", help="マスタの sheet 名")] = "Sheet1",
    output_path: Annotated[Path, typer.Option("--output", "-o", help="出力 Excel ファイル")] = Path(
        "output/join_result.xlsx"
    ),
    no_open: Annotated[bool, typer.Option("--no-open", help="完了後にファイルを自動で開かない")] = False,
) -> None:
    if len(csv_keys) != len(master_keys):
        raise typer.BadParameter("--csv-key と --master-key の数が一致していません。")

    df = pd.read_csv(csv_path, dtype=str)
    master = pd.read_excel(master_path, sheet_name=master_sheet, dtype=str)

    for col in csv_keys:
        df[col] = df[col].astype("string").str.strip()
    for col in master_keys:
        master[col] = master[col].astype("string").str.strip()

    result = df.merge(
        master,
        left_on=csv_keys,
        right_on=master_keys,
        how="outer",
        indicator=True,
        suffixes=("_csv", "_master"),
    )

    outputs = {
        key: result[result["_merge"] == meta["merge_value"]].drop(columns="_merge")
        for key, meta in SHEETS.items()
    }

    summary = pd.DataFrame(
        [
            {"type": "CSV", "sheet": "", "description": "元データCSVの件数", "count": len(df)},
            {"type": "MASTER", "sheet": "", "description": "マスタExcelの件数", "count": len(master)},
            *[
                {
                    "type": key,
                    "sheet": meta["sheet_name"],
                    "description": meta["description"],
                    "count": len(outputs[key]),
                }
                for key, meta in SHEETS.items()
            ],
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        write_sheet_as_table(writer, summary, "summary", "SummaryTable")
        for key, meta in SHEETS.items():
            write_sheet_as_table(writer, outputs[key], meta["sheet_name"], meta["table_name"])

    print("Join result counts")
    print(f"{'Type':<10} {'Count':>10}")
    print("-" * 22)
    print(f"{'CSV':<10} {len(df):>10}")
    print(f"{'MASTER':<10} {len(master):>10}")
    for key in SHEETS:
        print(f"{key:<10} {len(outputs[key]):>10}")
    print()
    print(f"Output: {output_path}")

    if not no_open:
        open_file(output_path)


if __name__ == "__main__":
    app()
