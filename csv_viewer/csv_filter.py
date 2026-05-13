from __future__ import annotations

import argparse
import csv
import configparser
import logging
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

from openpyxl import Workbook

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.ini"
OUTPUT_DIR = CONFIG_PATH.parent / "output"

NULL_LIKE = {"null", "none", "n/a", r"\n", "na"}
ENCODING_CANDIDATES = ["utf-8", "cp932"]


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"columns": [], "filters": {}, "file": None, "folder": None, "max_rows": None}

    parser = configparser.ConfigParser()
    parser.optionxform = str  # キー名の大文字小文字を保持
    parser.read(CONFIG_PATH, encoding="utf-8")

    default = parser["default"] if parser.has_section("default") else {}
    file_path: str | None = default.get("file") or None
    folder:    str | None = default.get("folder") or None
    max_rows:  int | None = int(default["display_rows"]) if default.get("display_rows") else None

    columns: list[str] = []
    if parser.has_section("columns"):
        raw = parser["columns"].get("names", "")
        columns = [c.strip() for c in raw.split(",") if c.strip()]

    filters: dict[str, str] = {}
    if parser.has_section("filter"):
        filters = dict(parser["filter"])

    return {"columns": columns, "filters": filters, "file": file_path, "folder": folder, "max_rows": max_rows}


def is_null_like(value: str) -> bool:
    return value.strip().lower() in NULL_LIKE


def display_value(value: str) -> str:
    if is_null_like(value):
        return "(null)"
    return value


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    for enc in ENCODING_CANDIDATES:
        try:
            with path.open(encoding=enc, newline="") as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
            logger.info("encoding: %s", enc)
            return headers, rows
        except (UnicodeDecodeError, LookupError):
            logger.warning("encoding '%s' failed, trying next", enc)
    raise SystemExit(f"failed to decode {path}: tried {ENCODING_CANDIDATES}")


_PAGE_SIZE = 5


def resolve_filter_columns(headers: list[str], filters: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    lower_to_original = {h.lower(): h for h in headers}
    for col_name, value in filters.items():
        if col_name in headers:
            resolved[col_name] = value
            continue
        all_matches = get_close_matches(col_name.lower(), lower_to_original, n=50, cutoff=0.4)
        if not all_matches:
            logger.warning("filter column '%s' not found, skipping", col_name)
            continue
        originals = [lower_to_original[c] for c in all_matches]
        offset = 0
        selected: str | None = None
        while True:
            page = originals[offset: offset + _PAGE_SIZE]
            print(f"\nfilter列 '{col_name}' が見つかりません。候補 ({offset + 1}-{offset + len(page)} / {len(originals)}):")
            for i, c in enumerate(page, 1):
                print(f"  {i}. {c}")
            has_more = offset + _PAGE_SIZE < len(originals)
            print(f"  {'m. 次の候補  ' if has_more else ''}0. スキップ")
            answer = input("番号を選択: ").strip()
            if answer == "0":
                break
            if has_more and answer == "m":
                offset += _PAGE_SIZE
                continue
            if answer.isdigit() and 1 <= int(answer) <= len(page):
                selected = page[int(answer) - 1]
                break
            print("  無効な入力です。もう一度入力してください。")
        if selected:
            resolved[selected] = value
    return resolved


def apply_filters(
    headers: list[str], rows: list[list[str]], filters: dict[str, str]
) -> list[list[str]]:
    for col_name, value in filters.items():
        if col_name not in headers:
            logger.warning("filter column '%s' not found, skipping", col_name)
            continue
        idx = headers.index(col_name)
        if value == "":
            # 空文字列 or null-like にマッチ
            rows = [r for r in rows if r[idx] == "" or is_null_like(r[idx])]
        else:
            rows = [r for r in rows if r[idx] == value]
    return rows


def fuzzy_match(name: str, candidates: list[str], cutoff: float = 0.6) -> str | None:
    lower_to_original = {c.lower(): c for c in candidates}
    matches = get_close_matches(name.lower(), list(lower_to_original), n=1, cutoff=cutoff)
    if matches:
        original = lower_to_original[matches[0]]
        logger.info("fuzzy match: '%s' -> '%s'", name, original)
        return original
    logger.warning("no match for '%s'", name)
    return None


def select_columns(
    headers: list[str], rows: list[list[str]], columns: list[str]
) -> tuple[list[str], list[list[str]]]:
    if not columns:
        return headers, rows

    indices: list[int] = []
    matched_headers: list[str] = []
    for col in columns:
        matched = fuzzy_match(col, headers)
        if matched is None:
            continue
        matched_headers.append(matched)
        indices.append(headers.index(matched))

    selected_rows = [[r[i] for i in indices] for r in rows]
    return matched_headers, selected_rows


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    formatted_rows = [[display_value(c) for c in r] for r in rows]
    all_rows = [headers] + formatted_rows

    if len(all_rows) == 1:
        return "(no data)"

    col_count = len(headers)
    widths = [max(len(r[i]) if i < len(r) else 0 for r in all_rows) for i in range(col_count)]

    lines = []
    for j, row in enumerate(all_rows):
        line = " | ".join(
            (row[i] if i < len(row) else "").ljust(widths[i]) for i in range(col_count)
        )
        lines.append(line)
        if j == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def write_excel(
    all_headers: list[str],
    headers: list[str],
    rows: list[list[str]],
    src_path: Path,
) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"{src_path.stem}_{ts}.xlsx"

    wb = Workbook()

    ws1 = wb.active
    ws1.title = "columns"
    ws1.sheet_view.showGridLines = False
    for col_name in sorted(all_headers):
        ws1.append([col_name])

    ws2 = wb.create_sheet("data")
    ws2.sheet_view.showGridLines = False
    ws2.append(headers)
    for row in rows:
        ws2.append(row)
    ws2.auto_filter.ref = ws2.dimensions

    wb.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV table viewer with fuzzy column selection")
    parser.add_argument("file", type=Path, nargs="?", help="path to .csv (overrides config.ini)")
    args = parser.parse_args()

    cfg = load_config()

    raw: Path | None = args.file or (Path(cfg["file"]) if cfg["file"] else None)
    if raw is None:
        parser.error("file not specified: pass as argument or set [default] file = ... in config.ini")

    if cfg["folder"]:
        raw_folder = Path(cfg["folder"])
        folder = raw_folder if raw_folder.is_absolute() else CONFIG_PATH.parent / raw_folder
    else:
        folder = None
    file_path = folder / raw if (folder and raw.parent == Path(".")) else raw

    all_headers, rows = read_csv(file_path)
    filters = resolve_filter_columns(all_headers, cfg["filters"])
    rows = apply_filters(all_headers, rows, filters)
    if not rows and filters:
        conditions = ", ".join(f"{k}={v!r}" for k, v in filters.items())
        print(f"該当なし: {conditions}")
        return
    headers, rows = select_columns(all_headers, rows, cfg["columns"])
    total = len(rows)
    display_rows = cfg["max_rows"]
    print(format_table(headers, rows[:display_rows] if display_rows else rows))
    summary = f"row count = {total}, column count = {len(headers)}"
    if display_rows is not None and total > display_rows:
        summary += f" (showing first {display_rows})"
    print(f"\n{summary}")

    out_path = write_excel(all_headers, headers, rows, file_path)
    print(f"Excel: {out_path}")


if __name__ == "__main__":
    main()
