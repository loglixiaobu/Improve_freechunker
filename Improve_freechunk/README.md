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

## 二、研究主张（不是"改进 FreeChunker"，而是把同一个故事讲到底）

> 完整论证见 [`IDEAS.md`](IDEAS.md)

FreeChunker 的故事：**chunking 不是找边界，是检索时选跨度。**

**我们的故事：`跨度`（span）这个词还留着最后一个假设 —— 证据是连续的。把它去掉。**

这个假设对 single-doc 成立，对 **multi-doc / multi-hop / long-icl** 不成立：
那里的证据天然**分散**，任何连续区间都装不下。

### 为什么这是"去掉限制"而不是"加模块"（taste 的落点）

FreeChunker 的注意力是 `softmax(QKᵀ/√d + P)·V`，`P ∈ {0,−∞}^{m×n}` 是 Chunk Pattern Mask。
论文里 P 是**固定带状**的 —— 但**数学上完全不需要带状**，
带状只是作者为了**枚举方便**做的选择（枚举所有 (g,s) 就能列出所有候选）。

> **带状是一个先验（prior），不是一个约束（constraint）。**

我们把 P 从预设对象变成**可学习对象**，让数据自己决定证据该有多分散，
而连续只是它的一种特例。于是得到一个**可解释的旋钮**：
正则调紧 → 应当**复现** FreeChunker（sanity check）；放松 → 观察长上下文域是否改善。

**而且这是论文自己写下的 future work**：

> "the chunk-pattern mechanism P_{g,s}[i,j] could in principle be extended to
> more complex **non-contiguous compositions**."

### 与已有工作的差异（必须说清，否则就是"换名字的 A"）

| 方法 | 检索单位 | 强制连续？ | 集合是单位？ |
|---|---|---|---|
| Proposition / 句子级检索 | 单个命题/句子 | — | ✗ 各句子**彼此独立** |
| FreeChunker | 跨度 | ✅ 强制 | ✅ |
| **本方法** | **证据集** | ✗ 任意 | ✅ |

### 三个组件（都是"证据集"这个表述的推论）

| 编号 | 名称 | 说明 |
|---|---|---|
| **C1** | **Learned Chunk Patterns** ★核心 | 把 P 从固定带状 → 可学习稀疏支持。**唯一改模型的部分** |
| **C2** | **Coverage-Aware Selection** | 检索单位是集合 → 选择目标应是**覆盖**而非排序；边际增益用全局索引**精确算**，贪心 **1−1/e**，零额外模型 |
| **C3** | **Set-Level Contrastive Training** | 组合族**天然提供最难负样本**（换掉一个句子只差一句）→ 监督信号免费 |

⚠️ **C2 单独拿出来是"通用集合选择"，不新鲜。它成立是因为 C1 把检索单位变成了集合** ——
是 C1 的推论，不是独立卖点。

### 先做诊断（便宜、可立刻做、且是论文 motivation）

**D3 是 go / no-go 闸门**：如果诊断显示 LongBench 的答案证据本来就集中在一处，
那"分散"这个 motivation 就是假的，**这个方向要重做，别硬上**。

### 明确不做（这些是"换名字的 A"）

BM25 混合检索、cross-encoder 重排、query expansion、粒度网格搜索、向量量化 ——
**口诀：如果一个改动换成固定粒度分块也照样能加，它就不属于这个故事。**


---

## 三、实验协议（所有改进都必须按这个跑）

**固定项（不许动）**

| 项     | 值                                                                                                                   |
| ----- | ------------------------------------------------------------------------------------------------------------------- |
| 数据集   | LongBench V2，503 题 / 6 域（single-doc 175、multi-doc 125、code-repo 50、long-dialogue 39、long-icl 81、long-structured 33） |
| 嵌入模型  | jina-embeddings-v2-small-en（项目内 `models/jina-embeddings-v2-small-en`）                                               |
| QA 模型 | Qwen3-8B @ vLLM，greedy（temperature=0）                                                                               |
| 检索指标  | Top-5 / Top-10 准确率                                                                                                  |
| 复现次数  | 1 次（greedy 下确定）                                                                                                     |

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

1. 先跑 **W1**（统计句子数 >4000 的样本数）—— 这关系到基线本身是否可信
2. 再上 **M1 · CASA** —— 主方法，纯后处理、零额外模型、有理论保证，最可能立刻见效
3. 然后 **M2 免重训部分**（粗到细检索，开销线）
4. 最后 **M3 + M2 训练部分**（重训，冲更强的论文）
