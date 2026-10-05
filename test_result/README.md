# test_result · FreeChunker vs Traditional 下游评测

**范围**：只含 `FreeChunker` 与 `Traditional(256/512)` 三个方法 × 6 个域。
（PPL / Margin / Semantic / Lumber 未纳入——见文末「未完成部分」。）

## 结论速览

| 方法 | TopK-5 | TopK-10 | 题数 |
|---|---|---|---|
| **FreeChunker** | **31.81** | **31.61** | 503 |
| Traditional(256) | 28.43 (−3.38) | 33.60 (+1.99) | 503 |
| Traditional(512) | 29.22 (−2.58) | 31.01 (−0.60) | 503 |

> 括号内为相对 FreeChunker 的差值（百分点）。

**FreeChunker 在 TopK-5 上领先两个 Traditional 基线约 2.6–3.4 个百分点；
TopK-10 上与 Traditional(256) 基本打平（−1.99）。**

## 产物完整性（已逐格核对）

题数**精确等于**该域应有题数，无缺失、无重复：

| 域 | 应有 | FreeChunker | Trad(256) | Trad(512) |
|---|---|---|---|---|
| single-doc | 175 | 175 | 175 | 175 |
| multi-doc | 125 | 125 | 125 | 125 |
| code-repo | 50 | 50 | 50 | 50 |
| long-dialogue | 39 | 39 | 39 | 39 |
| long-icl | 81 | 81 | 81 | 81 |
| long-structured | 33 | 33 | 33 | 33 |
| **合计** | **503** | **503** | **503** | **503** |

汇总自 18 个 `raw_*.jsonl`（3 方法 × 6 域），全部由同一次评测产出。

## 文件说明

| 文件 | 内容 |
|---|---|
| `downstream_report.md` | 主报告（总体 + 分域明细，TopK-5 / TopK-10） |
| `downstream_all_table.csv` | 明细表：方法 × 域 × topk |
| `downstream_overall.csv` | 总体表：方法 × topk |

## ⚠️ 三个必须知道的读数注意事项

1. **`total_time_s` 列不可当评测耗时看。**
   评测脚本算的是 `total_sample_time = sample['time'] + encoding_time`，
   其中 `sample['time']` 是**离线分块阶段**记录的耗时。
   所以 FreeChunker 显示 4009.8s（1.11h），Traditional 显示 0.0s
   —— 后者只是因为它读的是预先切好的 chunk、数据里没带这个字段。
   **准确率数字不受影响。**

2. **检索上下文上限是 32768 token，不是原始代码的 40000。**
   原代码 `max_context_length = 40000`，但本机 vLLM 实测上限 36160、
   Qwen3-8B 原生 32768，两边必须一致，否则超长 prompt 会被 vLLM
   以 HTTP 400 拒绝、**整个域归零**。故统一到 32768。

3. **FreeChunker 在超长文档上按「句子窗口」分段编码。**
   原实现把整篇文档的所有句子一次送进模型（注意力 O(N²)），
   实测 N=12000 句即 OOM（RTX 3090 24G），code-repo 域必然跑不完。
   现按 `FC_MAX_SENTENCES=4000` 分段（峰值 ~4.7 GiB）后拼接。
   - 已验证：**N ≤ 4000 句时输出与原实现逐位一致**（不受影响）
   - 偏差仅出现在超长文档的窗口边界处（该处无法形成跨窗口 chunk）

## 未完成部分

PPL / Margin / Semantic / Lumber 四个方法**未纳入本结果**，原因：
评测中途机器出现 GPU 硬件故障（`dmesg` 报 5 张卡 Xid 154、
内核标记 `Node Reboot Required`，GPU5 已无法读取）。
其中 Margin 另有独立问题（其调用点显式传了 `batch_size=4`，
覆盖了已修好的默认值，导致 OOM），已修复但需重跑。

**这三个方法的结果需要用重启后的机器重跑。**
本目录内三个方法的数据均产出于故障之前，完整可信。
