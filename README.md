# py-tools

A collection of small CLI utilities for data inspection.

## Install

```bash
pip install -e .
```

`csvview` / `xlview` がコマンドとして使えるようになり、どのフォルダからでも実行できる。

```bash
csvview data.csv
xlview book.xlsx
```

**`-e`（editable）を付けること。** 各ツールは自分のフォルダの `config.ini` を読むため、通常インストールだと設定ファイルが site-packages 内に埋まって編集できなくなる。`-e` ならリポジトリ内の `config.ini` をそのまま編集できる。

インストールせずに `python csv_viewer/show_table.py <file>` と直接実行することもできる（その場合はリポジトリルートを cwd にする必要がある）。

## Tools

| Tool | Command | Description |
|---|---|---|
| [csv_viewer](csv_viewer/README.md) | `csvview` | CSV をターミナルに整形表示。列選択（fuzzy match）・行 filter・列名一覧・フォルダ内の最新ファイル自動選択 |
| [excel_viewer](excel_viewer/README.md) | `xlview` | Excel（`.xlsx`）をターミナルに整形表示 |
