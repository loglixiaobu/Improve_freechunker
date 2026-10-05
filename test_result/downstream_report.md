# FreeChunker 下游评测 · 对照表

- 嵌入模型: jina-embeddings-v2-small-en（仅最小模型）
- QA 生成: Qwen3-8B @ vLLM
- 域: single-doc, multi-doc, code-repo, long-dialogue, long-icl, long-structured
- 汇总自 18 个 raw 文件

## 总体准确率

| 方法 | TopK-5 | TopK-10 | 题数 | 耗时 |
|---|---|---|---|---|
| FreeChunker | 31.81 | 31.61 | 503 | 1.11h |
| Traditional(256) | 28.43 (-3.38) | 33.60 (+1.99) | 503 | 0.00h |
| Traditional(512) | 29.22 (-2.58) | 31.01 (-0.60) | 503 | 0.00h |

> 括号内为相对 FreeChunker 的差值（百分点）

## 分域明细 · TopK-5

| 域 | FreeChunker | Traditional(256) | Traditional(512) |
|---|---|---|---|
| single-doc | 33.7 | 30.9 | 31.4 |
| multi-doc | 28.0 | 24.0 | 24.0 |
| code-repo | 48.0 | 44.0 | 42.0 |
| long-dialogue | 28.2 | 25.6 | 25.6 |
| long-icl | 25.9 | 27.2 | 27.2 |
| long-structured | 30.3 | 15.2 | 27.3 |

## 分域明细 · TopK-10

| 域 | FreeChunker | Traditional(256) | Traditional(512) |
|---|---|---|---|
| single-doc | 33.7 | 35.4 | 33.1 |
| multi-doc | 25.6 | 28.0 | 26.4 |
| code-repo | 46.0 | 52.0 | 52.0 |
| long-dialogue | 35.9 | 33.3 | 20.5 |
| long-icl | 28.4 | 30.9 | 29.6 |
| long-structured | 24.2 | 24.2 | 21.2 |
