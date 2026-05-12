# py_tools

A collection of small CLI utilities for data inspection.

---

## Tools

### csv_viewer — `csv_filter.py`

Display a CSV file as a formatted table, with optional column selection (fuzzy match) and row filtering.

**Usage**

```bash
# ファイルを直接指定
python csv_viewer/csv_filter.py <file.csv>

# config.ini に file を書いておけば引数なしで実行できる
python csv_viewer/csv_filter.py
```

CLI 引数が config.ini より優先される。

**config.ini**

```ini
[default]
encoding = utf-8
# file = data/sample.csv  # 省略時は CLI 引数が必須

[columns]
# Columns to display, comma-separated (fuzzy matched against headers)
names = name, date, amount

[filter]
# Exact column name = exact value
status = done
# Match empty string or null-like values (NULL, None, N/A, etc.)
category =
```

- `[default] file` — デフォルトの CSV パス。CLI 引数で上書き可能
- `[columns]` — omit to show all columns
- `[filter]` — omit to show all rows; leaving the value blank matches empty/null-like cells
- Null-like values are displayed as `(null)` in the output

**Dependencies**

None (stdlib only)

---

### excel_viewer — `show_table.py`

Display an Excel (`.xlsx`) sheet as a formatted table.

**Usage**

```bash
python excel_viewer/show_table.py <file.xlsx>
python excel_viewer/show_table.py <file.xlsx> -s <sheet_name>
```

**config.ini**

```ini
[default]
header_row = 1
# data_row = 2  # defaults to header_row + 1

# Per-sheet overrides
# [Sheet1]
# header_row = 3
# data_row = 5
```

**Dependencies**

```bash
pip install -r excel_viewer/requirements.txt
```
