# RAGFlow

[RAGFlow](https://github.com/infiniflow/ragflow)の環境構築手順をまとめたものです.

## 実行手順

1. RAGFlowをどこか別の場所にクローン

```
git clone https://github.com/infiniflow/ragflow.git
```

2. RAGFlowのディレクトリに移動

```
cd /path/to/ragflow
```

3. 動作確認済みのコミットをチェックアウト

```
git checkout c0799c53b32c4ab981c4fc46607adf2b2897d7cc
```

4. `docker/.env` を編集

- `TIMEZONE` の値を修正 (Asia/Tokyo)
- `VOLUME_DIR` の値を追加

```
VOLUME_DIR=/path/to/qllm/ragflow/volumes
```

- `ES_GROUP` の値を追加, `${VOLUME_DIR}/ragflow_elasticsearch` に対しての書き込み権限があるgidを設定

```
ES_GROUP=xxxxx
```

- `OLLAMA_PORT` の値を追加, 特に問題がなければ11434が良い

```
OLLAMA_PORT=11434
```

5. docker-composeファイルの修正

```
cp /path/to/qllm/ragflow/docker-compose-base.yml docker/docker-compose-base.yml
```

```
cp /path/to/qllm/ragflow/docker-compose.yml docker/docker-compose.yml
```

6. サービス起動

```
docker compose up
```

7. 必要なLLMをダウンロード

```
docker compose exec ollama ollama pull hf.co/rinna/qwen2.5-bakeneko-32b-instruct-gguf:Q8_0
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull llama3.3
```

8. ブラウザから80番ポートにアクセス

## 各種設定

### モデルの設定

[RAGFlowのマニュアル(Deploy LLM locally)](https://ragflow.io/docs/dev/deploy_local_llm#4-add-ollama)に従ってモデルの設定を行う.

5-1の手順で入力するModel type, Model nameはそれぞれ以下の通り.

- chat, hf.co/rinna/qwen2.5-bakeneko-32b-instruct-gguf:Q8_0
- embedding, bge-m3
- chat, llama3.3

5-2の手順で入力するbase URLは

```
http://ollama:11434
```

6の手順では, Chat modelにllama3.3, Embedding modelにbge-m3を設定する.

6まで進められれば一旦設定完了.

### ナレッジベースの設定

[RAGFlowのマニュアル(Configure knowledge base)](https://ragflow.io/docs/dev/configure_knowledge_base)に従ってナレッジベースの設定を行う.

RAPTORを使う場合のプロンプト例:

```
以下のパラグラフを簡潔に要約してください。それぞれのパラグラフは断片的な情報であり、元々一つの文章ではなかった可能性があることに留意してください。要約の際には数字に注意してください。また、事実を捏造しないようにも注意してください。

以下が要約するべきパラグラフです:
      {cluster_content}

以上が要約するべきパラグラフです。

要約:
```

## その他

### ナレッジベースに必要な情報が見つからなかったときにLLMの知識内から回答させる

Chat > Create an assistant > Chat Configuration > Prompt Engine > System からシステムプロンプトを以下のように変更する.

```
You are an intelligent assistant. Please summarize the content of the knowledge base to answer the question. Please list the data in the knowledge base and answer in detail. When all knowledge base content is irrelevant to the question, your answer must include the sentence "The answer you are looking for is not found in the knowledge base!" and generate answer using your internal knowledge rather than given knowledge base. Answers need to consider chat history.
      Here is the knowledge base:
      {knowledge}
      The above is the knowledge base.
```

デフォルトのシステムプロンプトに, 以下の一文を追加している.

and generate answer using your internal knowledge rather than given knowledge base.
