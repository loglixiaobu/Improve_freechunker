# 本次复现 vs 论文 · 差距表

**嵌入模型**：jina-embeddings-v2-small-en　**QA 生成**：Qwen3-8B @ vLLM　**基准**：LongBench V2

论文数值取自 FreeChunker.pdf 的 Table 1（jina-small-en 行）。差值 = 本次 − 论文，单位百分点。

## FreeChunker

（本次对应方法：`FreeChunker`）

| 域 | 论文 Top-5 | 本次 Top-5 | Δ | 论文 Top-10 | 本次 Top-10 | Δ |
|---|---|---|---|---|---|---|
| I. Single-Document QA | 35.43 | 33.71 | -1.72 | 32.57 | 33.71 | +1.14 |
| II. Multi-Document QA | 26.40 | 28.00 | +1.60 | 26.40 | 25.60 | -0.80 |
| III. Code Repository | 46.00 | 48.00 | +2.00 | 44.00 | 46.00 | +2.00 |
| IV. Long In-context Learning | 32.10 | 25.93 | -6.17 | 28.40 | 28.40 | +0.00 |
| V. Long-dialogue History | 28.21 | 28.21 | +0.00 | 23.08 | 35.90 | +12.82 |
| VI. Long Structured Data | 42.42 | 30.30 | -12.12 | 45.45 | 24.24 | -21.21 |
| **Overall** | **33.60** | **31.81** | **-1.79** | **31.61** | **31.61** | **+0.00** |

## Traditional

（本次对应方法：`Traditional(256)`）

| 域 | 论文 Top-5 | 本次 Top-5 | Δ | 论文 Top-10 | 本次 Top-10 | Δ |
|---|---|---|---|---|---|---|
| I. Single-Document QA | 30.86 | 30.86 | +0.00 | 35.43 | 35.43 | +0.00 |
| II. Multi-Document QA | 24.80 | 24.00 | -0.80 | 28.00 | 28.00 | +0.00 |
| III. Code Repository | 44.00 | 44.00 | +0.00 | 51.33 | 52.00 | +0.67 |
| IV. Long In-context Learning | 27.57 | 27.16 | -0.41 | 30.86 | 30.86 | +0.00 |
| V. Long-dialogue History | 26.50 | 25.64 | -0.86 | 33.33 | 33.33 | +0.00 |
| VI. Long Structured Data | 15.15 | 15.15 | +0.00 | 24.24 | 24.24 | +0.00 |
| **Overall** | **28.76** | **28.43** | **-0.33** | **33.53** | **33.60** | **+0.07** |

## 结论

1. **Traditional 几乎完全复现**：6 个域里有 2 个**逐位相同**（single-doc 30.86/35.43、long-structured 15.15/24.24），其余域的 Top-10 也基本吻合。
   → 说明**检索 + QA + 判分整条链路是忠实的**，FreeChunker 的差距不是评测流程造成的。

2. **FreeChunker 有系统性偏差**，集中在长上下文域（long-structured、long-icl）。

3. **已排除：granularity 配置不一致（我一开始的猜测是错的）。**
   论文正文写 *"the simplest combination of three granularities is directly preset: **(1, 2, 4)**"*，
   而代码里 `generate_shifted_matrix` 的默认值是 `[2, 4]`，看着像少了一档。
   但 `src/encoder.py` 里：
   ```python
   embedding = torch.cat([inputs_embeds, sequence_output], dim=1)
   ```
   `inputs_embeds` 就是 **Sentenizer 产出的原始句子向量 = granularity 1**，
   与模型产出的 g=2/g=4 chunk 向量拼接后一起进向量库。
   实测 N=179 句时 `grouped_texts = 179 + 267 = 446`（179 个单句 + 267 个多句 chunk）→ **配置本就是 (1,2,4)**。

4. **更可能的候选：我为绕开 OOM 加的「句子窗口」补丁。**
   `src/encoder.py` 现按 `FC_MAX_SENTENCES=4000` 分段编码。
   已验证 **N ≤ 4000 句时与原实现逐位一致**，
   但 N > 4000 的长文档会在窗口边界处**无法形成跨窗口 chunk**。
   而差距最大的 long-structured / long-icl 正是上下文最长的两个域 ——
   **这条更值得先查**：统计各域有多少样本的句子数超过 4000。

5. 其他待排查项：vLLM 上下文上限 32768（原代码 40000）对长上下文域的系统性影响。

## 本次未能对照的部分

论文 Table 1 还有 SemanticChunker / PPL / Margin / Lumber 四个方法，本次因 GPU 硬件故障未能跑完，暂无法对照。
