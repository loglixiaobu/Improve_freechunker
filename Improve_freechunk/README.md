# Improve_freechunk

目标：**在同一个数据集（LongBench V2，503 题 / 6 域）、同一套评测协议下，从准确率或开销上超越 FreeChunker。**

- 基线数字见 [`BASELINE.md`](BASELINE.md)
- 候选模块清单见 [`IDEAS.md`](IDEAS.md)
- 每个「加了模块且验证有效」的改动，都单独 git 提交并附详细说明（见文末《提交规范》）

---

## 一、FreeChunker 的核心创新点

论文原文见 `../FreeChunker.pdf`，实现见 `../src/`。以下按「为什么新」排序，而不是按论文顺序。

### 1. 范式转变：从「固定粒度分块」到「多粒度候选检索」

传统分块（含 Semantic / PPL / Margin / Lumber）都是**同一条流水线**：

```
切块（选一个粒度） → 每块单独编码 → 检索固定粒度的块
```

一旦切完，粒度就**锁死**了。想同时要「细粒度单句」和「粗粒度多句上下文」，
只能切两套索引、检索两次。

FreeChunker 换掉了第一步：**把句子当作原子单位**，
检索对象变成「文档中任意一段**连续句子跨度**」的候选集合。

> 论文原话："the framework treats sentences as atomic units and shifts from
> static chunk segmentation to flexible retrieval over configurable contiguous
> multi-granularity candidates."

关键：**粒度不再是切块时的决定，而是检索时的选择。**

### 2. Cross-Granularity Chunk Pattern：用一个掩码矩阵表达「所有粒度 × 所有位置」

对 n 个句子构造掩码 `P ∈ {0, −∞}^{m×n}`（m = 候选 chunk 数）：

```
P_{g,s}[i, j] = 0      if  s ≤ j < s + g
              = −∞     otherwise
```

即「从第 s 句开始、长度 g 句」的那一行全为 0，其余屏蔽。
遍历 (g, s) 就得到全部候选 —— 视觉上是一个**带状结构**（对角线附近有效）。

实现：`src/utils.py::generate_shifted_matrix`
（步长取 `max(1, g//2)`，即相邻候选 50% 重叠）。

### 3. Cross-Granularity Encoder：**一次前向**生成所有粒度的 chunk 向量

这是效率的全部来源。

朴素做法：对每个候选 `[s_i..s_j]` 拼接后单独过一遍编码器 → m 次前向，且句子被重复编码 m 遍。

FreeChunker 的做法：引入一个**可学习的 chunk 查询向量** `h_chk ∈ R^d`，
复制 m 份成 `H ∈ R^{m×d}`，把**句子向量 E 当 K/V** 做交叉注意力：

```
Q = W_Q·H          # 查询是「候选槽位」，不是文本
K = W_K·E          # 键是句子
V = W_V·E
Attn(H, E) = softmax( QKᵀ/√d + P ) · V     # P 就是上面那个掩码
```

- **Q 不是文本**，而是 m 个可学习槽位 → 一次前向并行产出 m 个 chunk 向量
- **P 作为加性掩码**决定每个槽位「看哪些句子」
- 句子向量 E 在所有粒度间**复用**，不重复编码

> 论文原话："This design allows parallel generation of all chunk embeddings in a
> single forward pass, with sentence-level embeddings being reused across
> different granularities, thereby avoiding redundant encoding of sentences."

实现：`src/freechunker.py::FreeChunkerModel.forward`
（`shifted_matrix` 构造掩码 → `encoder_attention_mask`）。

### 4. 训练方式：蒸馏式对齐（不是端到端检索训练）

- 数据：The Pile 采 100K 篇，每篇截断 8000 token 后切句
- 目标：让框架输出 `v` 逼近「真值」`e = M(concat(s_i..s_j))`，即**拼接后直接编码**的结果
- 粒度：训练用 **(2, 4, 8, 16, 32)** 句；**granularity 1 不训练**，直接用 Sentenizer 的原始句子向量
- 损失：余弦；论文 A800 上 >500 GPU 小时

⚠️ 一个容易误读的点：`generate_shifted_matrix` 默认 `granularities=[2,4]`，
但**候选集仍是 (1,2,4)** —— 因为 `src/encoder.py` 里

```python
embedding = torch.cat([inputs_embeds, sequence_output], dim=1)
#                            ↑ 原始句子向量 = granularity 1（免费）
```

`[2,4]` 只表示「模型要额外生成哪几档」。**配置与论文一致，别被默认值误导。**

### 5. 确定性的去重与拼接

不同粒度的候选必然重叠。重建上下文时：

1. 每个句子有全局编号 t，存为 `[Begin-t] S_t [End-t]`
2. 把检索到的 chunk 拆回原子句子
3. 同一 t 出现在多个候选里 → **保留得分最高的那个候选的副本**
4. 按 t 升序拼接；相邻 t 不连续处插 `"..."`

→ **结果与候选的物化顺序无关**，可复现。

实现：`src/aggregator.py`

### 6. 效率画像

论文 Figure 4：FreeChunker 与 Traditional 耗时相当，比复杂语义分块器**快最多 30×**。
原因有二：不做边界检测（省掉 LLM 推理），以及上面第 3 点的单次前向。

---

## 二、改进路线图（按「预期收益 / 实现成本」排序）

> 详细假设、验证方法、风险见 [`IDEAS.md`](IDEAS.md)

**第一批（纯检索层，不改模型、不需重训 —— 成本最低，先做）**

| 编号 | 方向 | 一句话 |
|---|---|---|
| I1 | **粒度感知的分数归一化** | 不同粒度的余弦分不可比，直接混排是次优的 |
| I2 | **混合检索（BM25 + 稠密，RRF 融合）** | 代码/结构化域靠精确匹配，稠密向量天然吃亏 |
| I3 | **token 预算下的覆盖优化** | 现在是「按分数取 top-k」，没考虑预算与冗余 |
| I4 | **候选剪枝** | 相邻候选 50% 重叠，向量库可大幅缩小（省内存/提速） |

**第二批（引入额外组件）**

| 编号 | 方向 | 一句话 |
|---|---|---|
| I5 | **重排（rerank）** | 召回 top-50 → cross-encoder 重排到 top-k |
| I6 | **查询扩展** | LLM 改写多子查询，分别检索后融合 |
| I7 | **粒度组合搜索** | 论文只用 (1,2,4)，试 (1,2,4,8) / 非均匀 / 分域自适应 |

**第三批（改模型 / 重训 —— 成本最高）**

| 编号 | 方向 | 一句话 |
|---|---|---|
| I8 | **对比学习训练信号** | 现在是余弦蒸馏（拟合真值向量），可加 hard-negative |
| I9 | **句子编码器微调** | Sentenizer 用的是原始 jina-small，未针对检索适配 |

**先要修掉的已知偏差（不属于"改进"，属于"对齐"）**

- **W1**：`FC_MAX_SENTENCES=4000` 的句子窗口 —— N>4000 的长文档在窗口边界无法形成跨窗口 chunk。
  已验证 N≤4000 时逐位一致；需统计有多少样本受影响。
- **W2**：vLLM 上下文上限 32768（原代码 40000），可能系统性压低长上下文域。

---

## 三、实验协议（所有改进都必须按这个跑）

**固定项（不许动）**

| 项 | 值 |
|---|---|
| 数据集 | LongBench V2，503 题 / 6 域（single-doc 175、multi-doc 125、code-repo 50、long-dialogue 39、long-icl 81、long-structured 33） |
| 嵌入模型 | jina-embeddings-v2-small-en（项目内 `models/jina-embeddings-v2-small-en`） |
| QA 模型 | Qwen3-8B @ vLLM，greedy（temperature=0） |
| 检索指标 | Top-5 / Top-10 准确率 |
| 复现次数 | 1 次（greedy 下确定） |

**主指标**：Overall Top-5、Overall Top-10（题数加权）
**副指标**：分域准确率、编码耗时、检索耗时、向量库条数/占用

**对照**：同时报告 `Traditional(256)`，因为它在本次复现中与论文**逐位吻合**，
是判断「本次改动是否有效」的**锚点** —— 如果 Traditional 也变了，说明改动泄漏到了不该影响的地方。

**判定"有效果"的门槛**（避免把噪声当收益）：

- Overall 提升 **≥ +0.5 个百分点**，且
- **至少 4/6 个域不变差**，且
- 若是开销方向的改进：向量库条数或耗时**降 ≥ 20%**，同时 Overall **下降 < 0.5 个百分点**

---

## 四、提交规范

**每个"加了模块且验证有效"的改动 = 一个独立 commit。** 提交信息必须包含：

```
<type>(<module-id>): <一句话结论>

## 动机
为什么做这个改动，针对基线的哪个具体短板。

## 改动
改了哪些文件、核心逻辑是什么（关键代码片段）。

## 实验
- 命令: <可复制的完整命令>
- 环境: <如有特殊依赖>
- 原始产物: Improve_freechunk/results/<exp-id>/

## 结果
| 指标 | 基线 | 本次 | Δ |
|---|---|---|---|
| Overall Top-5 | 31.81 | ... | ... |
| Overall Top-10 | 31.61 | ... | ... |
| Traditional(256) Top-5 | 28.43 | ... | ...(锚点校验) |
| 向量库条数 | ... | ... | ... |
| 编码耗时 | ... | ... | ... |

## 结论与后续
是否达到判定门槛；没达到也要提交（标记为 negative result），
并写清下一步假设。
```

**为什么连负面结果也提交**：避免同一个想法被重复尝试，也让"为什么最后选了这条路"有据可查。

---

## 五、目录约定

```
Improve_freechunk/
├── README.md           # 本文件：创新点 + 路线图 + 实验协议 + 提交规范
├── IDEAS.md            # 候选模块清单（假设 / 成本 / 验证方法 / 风险）
├── BASELINE.md         # 基线数字（本次实测 + 论文 + 差距）
├── modules/            # 新模块的代码（每个模块一个子目录）
├── scripts/
│   ├── eval_variant.sh # 评测入口（复用现有评测链路）
│   └── make_gap.py     # 生成「变体 vs 基线 vs 论文」对照表
└── results/            # 每次实验的原始产物（每次一个 <exp-id>/ 子目录）
```

---

## 六、当前阻塞

**机器 GPU 硬件故障，暂不能跑实验。** `dmesg` 报 5 张卡 Xid 154、
内核标记 `Node Reboot Required`，GPU5 已无法读取。**需重启后才能开始第一批实验。**

重启后按顺序：
1. 先跑 W1（统计句子数 >4000 的样本数）—— 这关系到基线本身是否可信
2. 再跑 I1（粒度感知分数归一化）—— 成本最低、最可能立刻见效
