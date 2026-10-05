# 文献调研：chunking / 不 chunking 的设计空间

> 调研日期 2026-10-05。**目的**：在做之前搞清楚我们要走的方向有没有被占。
> **方法**：先查 umbrella 综述 → 再中英文关键词分头检索 → 对候选前例抓原文核实。
> 凡只读到摘要、没读全文的，下文都标注了「仅摘要」。

---

## 一、结论先行

### 1.1 我们原定的三个组件，占据程度

| 组件 | 占据程度 | 最直接前例 | 剩下的空间 |
|---|---|---|---|
| **C1 · 可学习掩码**（P 从预设带状 → 可学习） | 🟡 **部分占据** | MoG（COLING 2025，在**预设菜单上路由**，不是学掩码） | 没找到直接学掩码的；但邻近区域拥挤 |
| **C2 · 覆盖感知选择**（次模 + 预算） | 🔴 **已被占据** | **arXiv 2607.00725**（预算约束次模证据打包，**允许非连续**）、SubMod-RAG | 几乎没有 —— 连"非连续"这一点都被做了 |
| **C3 · 集合级对比训练** | 🟡 **部分占据** | SetCSE（ICLR 2024，句子集合的对比学习 + 集合运算） | 有缝，但"集合嵌入"不是新概念 |
| **整体框子**（非连续证据集） | 🟡 **部分占据** | 2607.00725（非连续选取）、HyCE-RAG（分散证据）、CFIC（跳读解码） | 需要重新找差异化点 |

**一句话**：我们原定的故事**一半以上已经被做过**，尤其是最"实"的那一半（C2）。

### 1.2 ⚠️ 最致命的一条：读者规模会吞掉收益

arXiv 2607.00725 的核心实验（HotpotQA，3 seeds）：

| Reader | 次模打包 − 启发式 ΔF1 | p |
|---|---|---|
| 3B fp16 | **+0.022** | <0.05 |
| 7B fp16 | −0.010 | 0.45 |
| 14B 4-bit | **−0.029** | 0.013 |

**在 3B reader 上有效，7B 归零，14B 转负。** 作者自己解释为：
reader 足够强时，证据密度不再是瓶颈，精挑细选反而挤掉了别的信息。

**我们用的是 Qwen3-8B。** 也就是说：**即使我们把"证据集组装"做到完美，
在 8B reader 上很可能测不出提升。** 这条必须先解决，否则整个方向是空的。

### 1.3 性能位置

| 项 | 值 | 口径 |
|---|---|---|
| LongBench v2 公开 SOTA（直接长上下文，无检索） | **66.3%**（Qwen3.8 Max） | 与我们的 RAG 设定**不可直接相减** |
| FreeChunker 论文（jina-small-en，RAG） | 33.60 / 31.61 | 我们的直接对照口径 |
| **我们的 FreeChunker 复现** | **31.81 / 31.61** | 同上，已对齐 |
| Traditional(256) 论文 / 我们 | 28.76·33.53 / 28.43·33.60 | 锚点，逐位吻合 |

→ 我们的基线**没有落后**，这一点是好的。风险不在基线弱，而在**改进空间可能本来就没有**。

---

## 二、设计空间全图（按检索单位分类）

这是本次调研最有用的产出：**所有方法本质上是在回答"检索单位是什么"。**

| 类别 | 检索单位 | 代表方法 | 备注 |
|---|---|---|---|
| **固定粒度** | 定长块 | Fixed-size, Recursive Character | 最快（<1s），survey 实测最稳 |
| **经典话题分割** | 块（按话题边界） | **TextTiling (1997)**, EDTS（熵） | ⚠️ survey 指出 TextTiling **被现代 RAG 评测严重忽略** |
| **语义** | 块（按相似度跌落） | Semantic Chunking, Recursive Semantic | survey 里检索指标最好的之一 |
| **聚类** | 块（相邻句聚类） | Sequential HAC, Max-Min | 稳定但下游弱 |
| **结构/代码** | 按语法结构 | AutoChunker, **cAST**（AST）, S2（谱聚类） | 面向代码/结构化文档 |
| **LLM 判边界** | 块 | LumberChunker, HiChunk, LGMGC, Pseudo-Instruction, Meta-Chunking(PPL/Margin) | 贵；survey 实测**收益不成比例** |
| **命题级** | 命题（原子事实） | **DenseX / Propositions** | 更细粒度，但**失去"组合"** |
| **句子级** | 句子 | 句子窗口检索 | 同上 |
| **嵌入级** | 块（但带全文上下文） | **Late Chunking**（Jina 2024）, Contextual Retrieval（Anthropic） | 不换单位，换编码方式 |
| **层级** | 树/多级 | **RAPTOR**, Auto-merging Retrieval, HiChunk | 用层级替代单粒度 |
| **多向量** | token 级 | **ColBERT / ColBERTv2** | 迟交互，非连续匹配 |
| **免分块（生成式）** | 由解码器生成的证据文本 | **CFIC（ACL 2024）** | 用文档隐状态 + 约束前缀解码 + **跳读解码** |
| **免分块（图）** | 超边/子图 | **HyCE-RAG**（分散证据）、GraphRAG | 多跳社区 |
| **集合** | 句子集合 | **SetCSE**, Set-Theoretic Compositionality | 集合运算 / 组合性评测 |
| **上下文选择** | 已检索片段的子集 | **2607.00725**, SubMod-RAG | 次模 + 预算，**允许非连续** |

---

## 三、逐线核实（只列对我们最有威胁的）

### 3.1 🔴 arXiv 2607.00725 — *What Survives Into Context*（2026-07）

**做了什么**：把 reader 上下文构造表述为**预算约束的单调次模最大化**：

```
F(S) = w_rel·Rel(S) + w_qry·QueryCov(S) + w_cov·Repr(S) + w_div·Div(S)
s.t. cost(S) ≤ B
```

- 单位：**snippet**（从文档/段落里取的片段）
- **明确允许非连续**：用 `Div`（跨文档铺开）+ `Repr`（饱和设施选址）鼓励散开取
- 算法：按边际增益/代价比的贪心 + Lin–Bilmes 单例兜底（**标准模板**）
- 作者自己说：**"the optimizer is textbook"**，贡献在于把它用到 reader 上下文打包 + 四项目标 + 受控评测

**它自己承认的漏洞**（原文 scope）：
- 正面结果**只有一个数据集**（HotpotQA）、一个预算档（B=160）
- 单一模型家族（Qwen2.5）单一 embedder（bge-small）
- **reader 规模阶梯上：3B 有效、7B 归零、14B 转负**
- 诊断对长自由文本答案退化

**与我们 C2 的差异**：**几乎没有。** 我们的 C2 是"用句子级覆盖做次模选择"，
他们是"用片段级覆盖做次模选择"，且同样允许非连续。
→ **C2 不能作为贡献点。**

**唯一可利用的点**：他们用的是**现成片段**做集合选择；
**没有人把"非连续集合"做成编码器的输出**。这留给了 C1。

### 3.2 🟡 MoG — *Mix-of-Granularity*（COLING 2025）

**做了什么**：受 MoE 启发，训一个 **router 动态决定**该用哪个分块粒度。
**与我们 C1 的差异**：MoG 是在**预设的粒度菜单上做路由**；
C1 是**把掩码本身变成可学习的**（候选空间不由人预设）。
→ 不同。但"让粒度自适应查询"这个大方向已被占，不能再作为卖点。

### 3.3 🟡 CFIC — *Chunking-Free In-Context Retrieval*（ACL 2024）

**做了什么**：**完全绕开分块**。用文档的 encoded hidden states，
**自回归解码**出查询需要的证据文本；两个解码策略：
Constrained Sentence Prefix Decoding + **Skip Decoding**（跳读）。

**对我们的威胁**：这是"不 chunking"这条线最强的前例。
而且 **Skip Decoding 已经隐含了"非连续"** —— 它可以在生成证据时跳过中间句子。
→ 如果我们的卖点是"证据可以非连续"，CFIC 是一个很难绕的前例。
**必须读全文确认 Skip Decoding 到底能跳多少。**（本文档仅依据摘要，**待核实**）

### 3.4 🟡 HyCE-RAG（2026-07）

**做了什么**：超图（实体-关系-证据）建模，把**分散在多文档的证据**连成证据链。
**差异**：它是**图结构**路线，靠实体关系连边；
我们想的是**放宽 chunker 的连续性**。机制完全不同。
→ 不构成直接前例，但说明"分散证据"这个问题**已经有人在打**。

### 3.5 🟡 SetCSE（ICLR 2024）/ Set-Theoretic Compositionality（2025）

**做了什么**：SetCSE 用对比学习做**句子集合的运算**（交、差、运算序列）；
另一篇系统评测 7 种经典 + 9 种 LLM 句编码器的**集合组合性**。
**威胁**："句子集合可以被嵌入、可以做运算"**不是新想法**。
→ C3（集合级对比训练）需要重新找差异化。

### 3.6 经典方法：TextTiling（1997）值得重新看

综述明确写：TextTiling **"notably overlooked in modern RAG evaluations"**。
而且综述的总发现是：
> "More expensive methods did not yield meaningful effectiveness gains while adding substantial overhead."
> "Chunk quality and structural coherence matter more than chunk quantity."

→ 老方法可能非常能打 —— 这与你说的"老方法可能也非常有效"一致。

---

## 四、综述层面的确认（"没做过"的最强证据）

**arXiv 2606.00881**（*Chunking Methods on RAG*，Wrocław，自称首个系统性评测）：

逐节核对后，明确**没有**覆盖：

1. ❌ **非连续 / 分散证据检索** —— 无专门讨论。
   只顺带提了一句文学类数据集"relevant information may be distributed across distant parts of the text"。
2. ❌ **重叠候选的组装 / 去重 / 选择** —— 明确"left implicit rather than treated as a research problem"。
3. ⚠️ **命题级检索** —— 列为方法类别，但**未做实验**。

**它自己承认的漏洞**（这些是缺口所在）：
- 只用了**一个 retriever（bge-m3）+ 一个 reranker** → **chunking × retriever 的交互完全未知**
- 大多数数据集**没有 golden chunk**，相关性用答案 span 近似 → **评测口径本身不可靠**
- 许多方法因超时/内存直接失败，被排除在下游评测外
- **核心问题未解**：
  > "Whether chunking methods have a *meaningful* impact on RAG quality,
  > or whether observed differences are mainly driven by dataset selection
  > and implementation effects, remains open."

---

## 五、仍然开放的空间（剥掉被占据的部分之后）

按"能不能撑起一篇论文"排序：

### 🟢 空位 1：**分块到底在什么条件下才起作用？**（诊断 → 机制）
综述自己说这个问题"remains open"；2607.00725 又给出一个反直觉现象
（curation 收益随 reader 变强而消失）**但没解释机制**。
- 没有人系统性地回答：分块的收益**由什么变量决定**
  （reader 规模？检索器强度？证据分散度？题目类型？）
- 这可以做成**"分块的作用域地图"**，并且**能直接指导方法设计**
- 风险：偏分析论文，方法贡献弱

### 🟢 空位 2：**chunking × retriever 的交互**
综述只测了一个 retriever 就承认这是漏洞。
不同分块粒度与不同检索器（稠密 / 多向量 / 稀疏）**最优搭配可能完全不同**。
- 这解释了为什么各家结论互相矛盾
- 风险：偏实证，故事性弱

### 🟢 空位 3：**把"非连续集合"做成编码器的输出**（我们 C1 剩下的那部分）
2607.00725 是在**现成片段**上做非连续选择；
**没有人让编码器直接产出"非连续句子集"的表示**。
- 这是 C1 唯一还没被占的角落
- ⚠️ 但必须回答：**它比"现成片段的非连续选择"强在哪？**
  如果答案是"差不多"，那就没有贡献

### 🟡 空位 4：**经典话题分割的复兴**
TextTiling 被现代评测忽略。但"复兴老方法"本身不是故事，
除非能说清**为什么当年被放弃、而现在条件变了**。

---

## 六、最小验证实验（能证伪整个方向）

**在动任何模型之前，必须先做这两件事：**

### E1 · 分散性检验（决定 motivation 真假）
用现有产物（不动模型、不占 GPU）：统计 FreeChunker top-k 覆盖的句子
在文档中的**分布形态**。
- 若长期挤在一处 → "证据分散"这个 motivation 是假的 → **方向作废**
- 若经常散开 → motivation 成立

### E2 · 收益上界检验（决定值不值得做）
在**同样的 8B reader** 上，对比：
- 现状：按分数取 top-k
- 上界：同预算下**贪心最优**选择
- 对照：2607.00725 的次模打包

**判据**：如果"最优选择"相对"top-k"的提升 < 0.5pt，
则**整个组装方向在这个 reader 规模上就没有空间** —— 立刻停，别做。

> ⚠️ E2 必须在 8B reader 上做。2607.00725 的教训就是：
> 在 3B 上测出来的 +0.022，到 7B 就没了。

---

## 七、风险与审稿人会怎么打

| 拒稿理由 | 应对 |
|---|---|
| "次模上下文选择已被 2607.00725 做过" | **放弃 C2 作为贡献**，只在 related work 里引用 |
| "非连续检索 CFIC 已经做了" | 读 CFIC 全文，把差异做成**有数字**的核心论据 |
| "收益可能在 8B reader 上消失" | **必须先做 E2**；若上界不足 0.5pt 就别投 |
| "句子集合嵌入 SetCSE 做过" | C3 不作为独立贡献 |
| "只是把 MoG 的 router 换成 mask" | 说清"学掩码"与"在菜单上路由"的本质区别（候选空间是否预设） |

---

## 八、参考条目表

| # | 出处 | 与本思路的关系 |
|---|---|---|
| R1 | arXiv 2606.00881, *Chunking Methods on RAG*（2026-05） | **umbrella 综述**；确认非连续/组装未被覆盖 |
| R2 | arXiv 2607.00725, *What Survives Into Context*（2026-07） | **C2 的直接前例**；给出 reader 规模警告 |
| R3 | SubMod-RAG（novix.science，预印本，**未正式发表**） | C2 前例（次模 + 基数/背包约束） |
| R4 | MoG, COLING 2025（arXiv 2406.00456） | 自适应粒度的前例（菜单路由，非学掩码） |
| R5 | CFIC, ACL 2024（2024.acl-long.71） | **免分块前例**；Skip Decoding 可能已含非连续（**待读全文**） |
| R6 | Late Chunking, arXiv 2409.04701（Jina） | 嵌入级上下文；说明"上下文丢失"已被广泛关注 |
| R7 | DenseX, arXiv 2312.06648 | 命题级检索前例 |
| R8 | RAPTOR, arXiv 2401.18059 | 层级检索前例 |
| R9 | ColBERTv2, arXiv 2112.01488 | 多向量/迟交互前例 |
| R10 | SetCSE, ICLR 2024（OpenReview zEHGSN8Hy8） | 句子集合运算前例 |
| R11 | Set-Theoretic Compositionality, arXiv 2502.20975 | 集合组合性评测 |
| R12 | HyCE-RAG, arXiv 2607.22597 | 分散证据（图路线） |
| R13 | TextTiling, Hearst 1997 | 经典话题分割，**被现代评测忽略** |
| R14 | FreeChunker 论文 + 仓库 | 基线 |

---

## 九、下一步（待定，见回复）

本次调研的结论是：**原定方向的"实"的部分已被占据，且存在 reader 规模的硬风险。**
建议先做 E1/E2 两个廉价检验，再决定是否调整方向。
