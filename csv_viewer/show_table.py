from __future__ import annotations

import argparse
import csv
import configparser
import importlib.util
import logging
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

# openpyxl は Excel 書き出しでしか使わないので write_excel() 内で import する。
# 未インストールでも表示・列名一覧・filter は標準ライブラリだけで動く。

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.ini"
OUTPUT_DIR = CONFIG_PATH.parent / "output"

NULL_LIKE = {"null", "none", "n/a", r"\n", "na"}
ENCODING_CANDIDATES = ["utf-8", "cp932"]
LATEST_BY_CHOICES = ("mtime", "name")

_DEFAULT_CONFIG = {
    "columns": [], "filters": {}, "file": None, "folder": None,
    "max_rows": None, "latest_by": "mtime",
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return dict(_DEFAULT_CONFIG)

    # inline_comment_prefixes: `key = value  # コメント` を値の右側に書けるようにする。
    # `#` の直前に空白が要るので、`code = A#1` のような値はそのまま保持される。
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    parser.optionxform = str  # キー名の大文字小文字を保持
    parser.read(CONFIG_PATH, encoding="utf-8")

    default = parser["default"] if parser.has_section("default") else {}
    file_path: str | None = default.get("file") or None
    folder:    str | None = default.get("folder") or None
    max_rows: int | None = None
    if default.get("display_rows"):
        try:
            max_rows = int(default["display_rows"])
        except ValueError:
            raise SystemExit(f"invalid display_rows: {default['display_rows']!r} (expected an integer)")

    latest_by = (default.get("latest_by") or "mtime").strip().lower()
    if latest_by not in LATEST_BY_CHOICES:
        raise SystemExit(
            f"invalid latest_by: {latest_by!r} (expected one of {', '.join(LATEST_BY_CHOICES)})"
        )

    columns: list[str] = []
    if parser.has_section("columns"):
        raw = parser["columns"].get("names", "")
        columns = [c.strip() for c in raw.split(",") if c.strip()]

    filters: dict[str, str] = {}
    if parser.has_section("filter"):
        filters = dict(parser["filter"])

    return {
        "columns": columns, "filters": filters, "file": file_path, "folder": folder,
        "max_rows": max_rows, "latest_by": latest_by,
    }


def latest_csv(folder: Path, latest_by: str = "mtime") -> Path:
    """folder 直下で最も新しい .csv を返す。サブフォルダは見ない。"""
    if not folder.is_dir():
        raise SystemExit(f"folder not found: {folder}")

    candidates = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if not candidates:
        raise SystemExit(f"no .csv found in {folder}")

    if latest_by == "name":
        return max(candidates, key=lambda p: p.name)
    # 更新日時。同着はファイル名で決定的に選ぶ
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def resolve_file_path(
    raw: Path | None, folder: Path | None, latest_by: str
) -> tuple[Path, bool]:
    """読み込む CSV を決める。戻り値は (パス, 最新ファイルとして自動選択したか)。"""
    if raw is None:
        # file 未指定 → folder 内の最新 CSV（folder は呼び出し側で検証済み）
        assert folder is not None
        return latest_csv(folder, latest_by), True

    # filename only なら folder と結合。フルパスなら folder は無視
    path = folder / raw if (folder and raw.parent == Path(".")) else raw
    if path.is_dir():
        return latest_csv(path, latest_by), True
    return path, False


def is_null_like(value: str) -> bool:
    return value.strip().lower() in NULL_LIKE


def display_value(value: str) -> str:
    if is_null_like(value):
        return "(null)"
    return value


def read_csv(path: Path, headers_only: bool = False) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    for enc in ENCODING_CANDIDATES:
        try:
            with path.open(encoding=enc, newline="") as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                rows = [] if headers_only else list(reader)
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


def format_columns(headers: list[str]) -> str:
    if not headers:
        return "(no columns)"

    width = len(str(len(headers)))
    lines = [f"{i:>{width}}. {h}" for i, h in enumerate(headers, 1)]
    lines.append("")
    lines.append(f"column count = {len(headers)}")
    return "\n".join(lines)


def write_excel(
    all_headers: list[str],
    headers: list[str],
    rows: list[list[str]],
    src_path: Path,
) -> Path:
    """Excel を書き出してパスを返す。openpyxl が無ければ ImportError。"""
    from openpyxl import Workbook

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
    parser.add_argument(
        "file", type=Path, nargs="?",
        help="path to .csv, or a folder to read its newest .csv (overrides config.ini)",
    )
    parser.add_argument(
        "-l", "--list-columns",
        action="store_true",
        help="列名だけを表示して終了する（filter・Excel 出力は行わない）",
    )
    args = parser.parse_args()

    cfg = load_config()

    if cfg["folder"]:
        raw_folder = Path(cfg["folder"])
        folder = raw_folder if raw_folder.is_absolute() else CONFIG_PATH.parent / raw_folder
    else:
        folder = None

    raw: Path | None = args.file or (Path(cfg["file"]) if cfg["file"] else None)
    if raw is None and folder is None:
        parser.error(
            "file not specified: pass as argument or set [default] file = ... "
            "(or folder = ... to use its newest .csv) in config.ini"
        )

    file_path, auto_selected = resolve_file_path(raw, folder, cfg["latest_by"])
    if auto_selected:
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        print(f"selected: {file_path}  ({mtime:%Y-%m-%d %H:%M})")

    if args.list_columns:
        all_headers, _ = read_csv(file_path, headers_only=True)
        print(format_columns(all_headers))
        return

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

    # 表示は成功しているので、openpyxl が無くても Excel だけ諦めて終わる。
    # find_spec で存在だけ確認する（try/except ImportError だと openpyxl 内部の
    # ImportError まで「未インストール」と誤って報告してしまう）
    if importlib.util.find_spec("openpyxl") is None:
        print("Excel: skipped — openpyxl が未インストール（pip install openpyxl で有効になる）")
    else:
        print(f"Excel: {write_excel(all_headers, headers, rows, file_path)}")


if __name__ == "__main__":
    main()
