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
folder = C:\Users\you\data   # filename only 指定時の base dir
file = sample.csv            # filename only → folder と結合。フルパスなら folder を無視
display_rows = 30            # 表示する最大行数。省略時は全行

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
| `[default] folder` | filename only 指定時の base directory |
| `[default] file` | デフォルトの CSV パス。CLI 引数で上書き可能 |
| `[default] display_rows` | 表示する最大行数。省略時は全行表示 |
| `[columns] names` | 表示列をカンマ区切りで指定。省略時は全列表示 |
| `[filter]` | 行の絞り込み条件。省略時は全行表示 |

文字コードは設定不要で、`utf-8` → `cp932` の順に自動判定する。

### パス解決の優先順位

1. CLI 引数
2. `config.ini` の `file`
3. どちらもなければエラー

指定されたパスが filename only の場合、`folder` と結合する。フルパスなら `folder` は無視。

### null-like の扱い

`NULL` / `None` / `N/A` / `NA` / `\n` は `(null)` と表示される。  
`[filter]` で値を空にすると、空文字列または null-like な行だけを残す。

### filter 列名が見つからない場合

`[filter]` の列名がヘッダーに完全一致しないとき、候補を5件ずつ表示して番号入力を求める。

```
filter列 'sttus' が見つかりません。候補 (1-5 / 7):
  1. status
  2. status_code
  ...
  m. 次の候補  0. スキップ
番号を選択:
```

`m` で次のページ、`0` でその条件をスキップ（絞り込みなしで続行）。

対話プロンプトなので、バッチ実行や CI では入力待ちで停止する。無人実行するなら `[filter]` の列名はヘッダーと完全一致させておく。

なお `[columns] names` の方は対話なしで、自動的に最も近い列名へ fuzzy match する（一致しなければその列を無視）。

## Dependencies

```bash
pip install -r csv_viewer/requirements.txt
```

表示結果は `output/` 配下に Excel（`columns` シート + `data` シート、auto-filter 付き）としても書き出される。
