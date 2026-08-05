# csv_viewer

CSV ファイルをターミナルに整形表示する CLI ツール。

## Usage

```bash
# ファイルを直接指定
python csv_viewer/csv_filter.py <file.csv>

# フォルダを指定 → その中で1番新しい .csv を自動で選ぶ
python csv_viewer/csv_filter.py <folder>

# config.ini に file を書いておけば引数なしで実行できる
python csv_viewer/csv_filter.py

# Windows — csv_view.bat（repo root）を使う場合
csv_view.bat
```

CLI 引数が config.ini より優先される。

### 列名だけを表示する

```bash
python csv_viewer/csv_filter.py --list-columns <file.csv>
python csv_viewer/csv_filter.py -l <file.csv>

# Windows
csv_columns.bat <file.csv>
```

```
1. name
2. date
3. amount

column count = 3
```

ヘッダー行だけを読んで連番付きで表示し、終了する。`[columns]` / `[filter]` の設定は無視され、Excel も出力しない。`[filter]` に書く列名を確認したいときや、巨大な CSV の中身を開かずに構造だけ見たいときに使う。

## config.ini

```ini
[default]
folder = C:\Users\you\data   # filename only 指定時の base dir
file = sample.csv            # filename only → folder と結合。フルパスなら folder を無視
                             # 空にすると folder 内で1番新しい .csv を自動選択
display_rows = 30            # 表示する最大行数。省略時は全行
latest_by = mtime            # 自動選択の基準。mtime（デフォルト） or name

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
| `[default] latest_by` | 最新ファイル自動選択の基準。`mtime`（デフォルト） or `name` |
| `[columns] names` | 表示列をカンマ区切りで指定。省略時は全列表示 |
| `[filter]` | 行の絞り込み条件。省略時は全行表示 |

文字コードは設定不要で、`utf-8` → `cp932` の順に自動判定する。

### パス解決の優先順位

1. CLI 引数
2. `config.ini` の `file`
3. どちらも無く `folder` があれば、その中で1番新しい `.csv`
4. すべて無ければエラー

指定されたパスが filename only の場合、`folder` と結合する。フルパスなら `folder` は無視。

### 最新ファイルの自動選択

指定先がフォルダのとき（CLI 引数がフォルダ／`file` が空で `folder` のみ指定）、その中で1番新しい `.csv` を選ぶ。

```bash
$ python csv_viewer/csv_filter.py C:\Users\you\data
selected: C:\Users\you\data\sales_0805.csv  (2026-08-05 09:12)
...
```

どのファイルを開いたかは必ず `selected:` 行に出る。

| 挙動 | 詳細 |
|---|---|
| 基準 | `latest_by = mtime`（更新日時、デフォルト）。同着はファイル名で決定的に選ぶ |
| | `latest_by = name` ならファイル名の降順で先頭。`売上_20260805.csv` のように名前へ日付が入る運用向け |
| 対象 | フォルダ直下の `.csv` のみ。**サブフォルダは見ない**。拡張子は大文字 `.CSV` も対象 |
| `.csv` が0件 | `no .csv found in <folder>` で終了 |

更新日時はダウンロード直後なら正確だが、コピーや zip 展開で元の日時が保たれないと崩れる。その場合は `latest_by = name` の方が安定する。

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
