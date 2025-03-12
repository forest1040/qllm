# Research Paper Report Generation

指定したトピックに関するarXivの論文をまとめたレポートを作成できます.

## ディレクトリ構成

- examples: レポートのアウトラインや設定ファイルの例が入っています
- results: 作成したレポートと中間ファイルを格納するディレクトリです(変更可)
- ollama: ollamaが使用するファイルを格納するディレクトリです(変更可)
- python_packages: Pythonの依存パッケージを格納するディレクトリです

## 実行手順(初回)

1. リポジトリをクローン

2. Research Paper Report Generationのディレクトリに移動

```
cd path/to/qllm/research_paper_report_generation
```

3. Ollamaの準備

  - 3.1. 環境変数ファイルをコピー

  ```
  cp examples/example.env .env
  ```

  デフォルトではollamaが使用するファイルはollamaディレクトリに格納されますが, 変更したい場合は.envファイルを編集してください.

  - 3.2. DockerからOllamaを起動

  ```
  docker compose up -d ollama
  ```

  - 3.3. 必要なLLMをダウンロード

  ```
  docker compose exec ollama ollama pull hf.co/rinna/qwen2.5-bakeneko-32b-instruct-gguf:Q8_0
  docker compose exec ollama ollama pull bge-m3
  docker compose exec ollama ollama pull llama3.3
  ```

4. 実行環境の準備

  - 4.1. Dockerからworkerサービスを起動

  ```
  docker compose up -d worker
  ```

  - 4.2. 依存パッケージをインストール

  ```
  docker compose exec pip install -t python_packages -r requirements.txt
  ```

  - 4.3. 設定ファイルをコピー

  ```
  cp examples/config.example.yml config.yml
  ```

  内容については[設定ファイル](#設定ファイル)を参照してください.

  - 4.4. アウトラインファイルをコピー

  ```
  cp examples/outlines/outline_quant_ph.md ./outline.md
  ```

  内容については[アウトラインファイル](#アウトラインファイル)を参照してください.

5. 実行

```
docker compose exec worker python main.py
```

デバッグメッセージを表示したい場合:

```
docker compose exec worker python main.py --debug
```

出来上がったレポートは, 設定ファイルで指定したwork_dirディレクトリにreport.mdとして出力されます.

6. 片づけ

```
docker compose down
```

## 実行手順(初回以降)

1. Research Paper Report Generationのディレクトリに移動

```
cd path/to/qllm/research_paper_report_generation
```

2. サービス起動

```
docker compose up -d
```

3. config.ymlを必要に応じて編集, 特にwork_dirの変更忘れに注意

4. レポート作成

```
docker compose exec worker python main.py
```

5. 片づけ

```
docker compose down
```

## プログラムの流れ

このプログラムは DOWNLOAD, INDEX, SC(Section Contents), REPORT の4フェーズから構成されています. それぞれのフェーズ完了時には中間ファイルが作られ, その時点から再開することができます.

- DOWNLOAD: arXivから論文をダウンロードするフェーズ
- INDEX: 論文ファイルをパースしてベクトルストアインデックスを作成するフェーズ
- SC: LLMを使ってレポートの本文を生成するフェーズ
- REPORT: SCの内容をまとめてレポートとして成形するフェーズ. 実行の都合上, IntroductionとConclusionの内容はSCではなくここで生成されます

## 設定ファイル

- phases: 実行するフェーズを指定します. 1フェーズずつ経過を確認して実行したい場合や, 中間ファイルから再開したい場合に活用してください
- paper_ids: arXivからダウンロードするpdfファイルのIDを直接指定します. これを設定した場合, `scirate`, `target_date`, `topics`, `num_papers_per_topic` の項目は無視されます
- scirate: SciRateを参照して評価の高い論文を取得します
- target_date: SciRateから検索する際の対象となる日付を設定します. scirateがtrueの場合のみ有効です
- topics: 論文を検索する分野を指定します
- num_papers_per_topic: 取得する論文の数を設定します. topicsで指定した1分野ごとに, この数の論文を取得します
- outline_file_path: 使用するアウトラインファイルへのパスを指定します. 相対パスを指定した場合, main.py が存在するディレクトリからの相対パスと解釈します
- work_dir: 作成したレポートと中間ファイルを格納するディレクトリを指定します. 相対パスを指定した場合, main.py が存在するディレクトリからの相対パスと解釈します
- max_retry: SCフェーズでLLMがイマイチな内容を出力した際に最大何回までやり直すかを設定します. 設定回数を超えて失敗した場合にはエラー終了します
- llm_model_name: レポートの内容を生成するモデル名を指定します
- embedding_model_name: indexingやretrievalに使用するembeddingモデル名を指定します
- evaluator_model_name: SCフェーズで生成した内容を評価するモデル名を指定します
- ollama_base_url: OllamaのURLを設定します
- evaluator_guideline: evaluatorの評価基準を設定します
- section_content_query_instruction: SCフェーズで内容生成のためのプロンプトを生成する際の指示を設定します
- section_content_answer_instruction: SCフェーズで内容を生成する際の指示を設定します
- introduction_instruction: Instructionを生成する際の指示を設定します
- conclusion_instruction: Conclusionを生成する際の指示を設定します

## アウトラインファイル

作成するレポートのアウトラインのmdファイルです. 以下の制約に従っている必要があります.

- 1行目にタイトルがあること
- 最初のセクションが Introduction であること
- 最後のセクションが Conclusion であること
- Latest Papers というタイトルのセクションを含むこと

