# csv_viewer

CSV ファイルをターミナルに整形表示する CLI ツール。

## Usage

```bash
# ファイルを直接指定
python csv_viewer/csv_filter.py <file.csv>

# config.ini に file を書いておけば引数なしで実行できる
python csv_viewer/csv_filter.py

# Windows — csv_view.bat（repo root）を使う場合
csv_view.bat
```

CLI 引数が config.ini より優先される。

## config.ini

```ini
[default]
encoding = utf-8
folder = C:\Users\you\data   # filename only 指定時の base dir
file = sample.csv            # filename only → folder と結合。フルパスなら folder を無視

[columns]
# 表示する列名をカンマ区切りで指定（fuzzy match）
names = name, date, amount

[filter]
# 列名 = 値（完全一致）
status = done
# 空欄 or null-like（NULL, None, N/A 等）にマッチ
category =
```

| 設定 | 説明 |
|---|---|
| `[default] encoding` | CSV の文字コード（デフォルト: `utf-8`） |
| `[default] folder` | filename only 指定時の base directory |
| `[default] file` | デフォルトの CSV パス。CLI 引数で上書き可能 |
| `[columns] names` | 表示列をカンマ区切りで指定。省略時は全列表示 |
| `[filter]` | 行の絞り込み条件。省略時は全行表示 |

### パス解決の優先順位

1. CLI 引数
2. `config.ini` の `file`
3. どちらもなければエラー

指定されたパスが filename only の場合、`folder` と結合する。フルパスなら `folder` は無視。

### null-like の扱い

`NULL` / `None` / `N/A` / `NA` / `\n` は `(null)` と表示される。  
`[filter]` で値を空にすると、空文字列または null-like な行だけを残す。

## Dependencies

```bash
pip install -r csv_viewer/requirements.txt
```

表示結果は `output/` 配下に Excel（`columns` シート + `data` シート、auto-filter 付き）としても書き出される。
