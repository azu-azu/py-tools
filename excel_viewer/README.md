# excel_viewer

Excel（`.xlsx`）をターミナルに整形表示する CLI ツール。

## Usage

```bash
# active sheet を表示
python excel_viewer/show_table.py <file.xlsx>

# シートを指定
python excel_viewer/show_table.py <file.xlsx> -s <sheet_name>
```

## config.ini

```ini
[default]
header_row = 1
# data_row = 2  # 省略時は header_row + 1

# シートごとに上書き
# [Sheet1]
# header_row = 3
# data_row = 5
```

| 設定 | 説明 |
|---|---|
| `header_row` | ヘッダー行の行番号（1-based） |
| `data_row` | データ開始行の行番号（1-based）。省略時は `header_row + 1` |

シート名と同名のセクションを書くと、そのシートだけ設定を上書きできる。

## Dependencies

```bash
pip install -r excel_viewer/requirements.txt
```
