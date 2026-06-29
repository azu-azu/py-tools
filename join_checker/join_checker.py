import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

EXCEL_PREVIEW_ROWS = 10_000

SHEETS = {
    "L": {
        "sheet_name": "L_left_only",
        "description": "CSVにだけ存在",
        "merge_value": "left_only",
    },
    "J": {
        "sheet_name": "J_joined",
        "description": "CSVとExcelマスタの両方に存在",
        "merge_value": "both",
    },
    "R": {
        "sheet_name": "R_right_only",
        "description": "Excelマスタにだけ存在",
        "merge_value": "right_only",
    },
}


def make_unique_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Excel はカラム名の大文字小文字を区別しないため、重複を _1, _2 ... でリネームする。"""
    seen: dict[str, int] = {}
    new_columns = []

    for col in data.columns:
        base = str(col).strip() or "Column"
        key = base.lower()
        count = seen.get(key, 0)
        new_columns.append(base if count == 0 else f"{base}_{count}")
        seen[key] = count + 1

    result = data.copy()
    result.columns = pd.Index(new_columns)
    return result


def reorder_columns(
    result: pd.DataFrame,
    csv_keys: list[str],
    master_keys: list[str],
    csv_columns: list[str],
    master_columns: list[str],
) -> pd.DataFrame:
    """join 後のカラム順を [csv_keys, master_keys, master_other, csv_other, rest] に並べ直す。"""
    # 両側に同名カラムがあると pandas が _csv / _master suffix を付ける
    conflict = (set(csv_columns) & set(master_columns)) - set(csv_keys) - set(master_keys)

    def as_csv(col: str) -> str:
        return f"{col}_csv" if col in conflict else col

    def as_master(col: str) -> str:
        return f"{col}_master" if col in conflict else col

    ordered: list[str] = []
    seen: set[str] = set()

    def add(col: str) -> None:
        if col in result.columns and col not in seen:
            ordered.append(col)
            seen.add(col)

    for col in csv_keys:
        add(col)
    for col in master_keys:
        add(col)
    for col in master_columns:
        if col not in master_keys:
            add(as_master(col))
    for col in csv_columns:
        if col not in csv_keys:
            add(as_csv(col))
    for col in result.columns:
        if col != "_merge":
            add(col)
    if "_merge" in result.columns:
        ordered.append("_merge")

    return result[ordered]


def create_columns_sheet(
    result: pd.DataFrame,
    csv_keys: list[str],
    master_keys: list[str],
    csv_columns: list[str],
    master_columns: list[str],
) -> pd.DataFrame:
    """result の各カラムがどのソースに由来するかを一覧化するメタデータ DataFrame を返す。"""
    rows = []

    for no, col in enumerate(result.columns, start=1):
        if col == "_merge":
            source, source_column = "SYSTEM", "_merge"
        elif col.endswith("_csv") and col[:-4] in csv_columns:
            source, source_column = "CSV", col[:-4]
        elif col.endswith("_master") and col[:-7] in master_columns:
            source, source_column = "MASTER", col[:-7]
        elif col in csv_keys:
            source, source_column = "CSV_KEY", col
        elif col in master_keys:
            source, source_column = "MASTER_KEY", col
        elif col in master_columns:
            source, source_column = "MASTER", col
        elif col in csv_columns:
            source, source_column = "CSV", col
        else:
            source, source_column = "UNKNOWN", col

        rows.append({"no": no, "join_column": col, "source": source, "source_column": source_column})

    return pd.DataFrame(rows)


def write_sheet(
    writer: pd.ExcelWriter,
    data: pd.DataFrame,
    sheet_name: str,
    table_name: str | None = None,
    use_table: bool = False,
) -> None:
    """DataFrame を Excel sheet に書き出す。use_table=True で Excel テーブル、False で autofilter。"""
    output_df = make_unique_columns(data)
    output_df.to_excel(writer, sheet_name=sheet_name, index=False)

    worksheet = writer.sheets[sheet_name]
    worksheet.hide_gridlines(2)

    rows, cols = output_df.shape
    if cols == 0:
        return

    if use_table:
        columns = [{"header": str(col)} for col in output_df.columns]
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
    else:
        worksheet.autofilter(0, 0, rows, cols - 1)

    # summary sheet: CSV / MASTER の合計行を黄色ハイライト
    if sheet_name == "summary":
        highlight = writer.book.add_format({"bold": True, "bg_color": "#FFFF00"})  # type: ignore[attr-defined]
        for row in (1, 2):
            for col in range(cols):
                worksheet.write(row, col, output_df.iat[row - 1, col], highlight)

    for col_idx, col_name in enumerate(output_df.columns):
        values = output_df.iloc[:, col_idx].dropna().head(1000)
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

    csv_columns = df.columns.tolist()
    master_columns = master.columns.tolist()

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

    result = reorder_columns(result, csv_keys, master_keys, csv_columns, master_columns)

    outputs = {
        key: result[result["_merge"] == meta["merge_value"]].drop(columns="_merge")
        for key, meta in SHEETS.items()
    }

    columns_df = create_columns_sheet(result, csv_keys, master_keys, csv_columns, master_columns)

    summary = pd.DataFrame(
        [
            {
                "type": "CSV",
                "sheet": "",
                "description": "元データCSVの件数",
                "count": len(df),
                "excel_preview_count": "",
                "csv_file": "",
            },
            {
                "type": "MASTER",
                "sheet": "",
                "description": "マスタExcelの件数",
                "count": len(master),
                "excel_preview_count": "",
                "csv_file": "",
            },
            *[
                {
                    "type": key,
                    "sheet": meta["sheet_name"],
                    "description": meta["description"],
                    "count": len(outputs[key]),
                    "excel_preview_count": min(len(outputs[key]), EXCEL_PREVIEW_ROWS),
                    "csv_file": f"{meta['sheet_name']}_full.csv",
                }
                for key, meta in SHEETS.items()
            ],
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        write_sheet(writer, summary, "summary", table_name="SummaryTable", use_table=True)
        write_sheet(writer, columns_df, "columns")

        for key, meta in SHEETS.items():
            full_csv_path = output_path.parent / f"{meta['sheet_name']}_full.csv"
            outputs[key].to_csv(full_csv_path, index=False, encoding="utf-8-sig")

            preview = outputs[key].head(EXCEL_PREVIEW_ROWS)
            write_sheet(writer, preview, meta["sheet_name"])

    print("Join result counts")
    print(f"{'Type':<10} {'Count':>12} {'Excel':>12} {'CSV':<30}")
    print("-" * 70)
    print(f"{'CSV':<10} {len(df):>12,} {'':>12} {'':<30}")
    print(f"{'MASTER':<10} {len(master):>12,} {'':>12} {'':<30}")
    for key, meta in SHEETS.items():
        count = len(outputs[key])
        preview_count = min(count, EXCEL_PREVIEW_ROWS)
        csv_file = f"{meta['sheet_name']}_full.csv"
        print(f"{key:<10} {count:>12,} {preview_count:>12,} {csv_file:<30}")
    print()
    print(f"Output Excel: {output_path}")
    print(f"Preview rows per sheet: {EXCEL_PREVIEW_ROWS:,}")

    if not no_open:
        open_file(output_path)


if __name__ == "__main__":
    app()
