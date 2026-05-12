from __future__ import annotations

import argparse
import csv
import configparser
import logging
from difflib import get_close_matches
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.ini"

NULL_LIKE = {"null", "none", "n/a", r"\n", "na"}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"encoding": "utf-8", "columns": [], "filters": {}}

    parser = configparser.ConfigParser()
    parser.optionxform = str  # キー名の大文字小文字を保持
    parser.read(CONFIG_PATH, encoding="utf-8")

    encoding = "utf-8"
    if parser.has_section("default"):
        encoding = parser["default"].get("encoding", "utf-8")

    file_path: str | None = None
    if parser.has_section("default"):
        file_path = parser["default"].get("file") or None

    folder: str | None = None
    if parser.has_section("default"):
        folder = parser["default"].get("folder") or None

    columns: list[str] = []
    if parser.has_section("columns"):
        raw = parser["columns"].get("names", "")
        columns = [c.strip() for c in raw.split(",") if c.strip()]

    filters: dict[str, str] = {}
    if parser.has_section("filter"):
        filters = dict(parser["filter"])

    return {"encoding": encoding, "columns": columns, "filters": filters, "file": file_path, "folder": folder}


def is_null_like(value: str) -> bool:
    return value.strip().lower() in NULL_LIKE


def display_value(value: str) -> str:
    if is_null_like(value):
        return "(null)"
    return value


def read_csv(path: Path, encoding: str = "utf-8") -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    with path.open(encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
    return headers, rows


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
    matches = get_close_matches(name, candidates, n=1, cutoff=cutoff)
    if matches:
        logger.info("fuzzy match: '%s' -> '%s'", name, matches[0])
        return matches[0]
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
    display_rows = [[display_value(c) for c in r] for r in rows]
    all_rows = [headers] + display_rows

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


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV table viewer with fuzzy column selection")
    parser.add_argument("file", type=Path, nargs="?", help="path to .csv (overrides config.ini)")
    args = parser.parse_args()

    cfg = load_config()

    raw: Path | None = args.file or (Path(cfg["file"]) if cfg["file"] else None)
    if raw is None:
        parser.error("file not specified: pass as argument or set [default] file = ... in config.ini")

    folder = Path(cfg["folder"]) if cfg["folder"] else None
    file_path = folder / raw if (folder and raw.parent == Path(".")) else raw

    headers, rows = read_csv(file_path, cfg["encoding"])
    rows = apply_filters(headers, rows, cfg["filters"])
    headers, rows = select_columns(headers, rows, cfg["columns"])
    print(format_table(headers, rows))


if __name__ == "__main__":
    main()
