#!/usr/bin/env python3
"""给 src/encoder.py 的 `encode()` 加句子窗口，避免 FreeChunker 在大文档上 OOM。

问题（2026-10-05 实测）
----------------------
原实现把**整篇文档的所有句子**一次性送进模型：
    outputs = self.model(inputs_embeds=[1, N, 1024], granularities=(2,4))
注意力是 O(N²)。实测（RTX 3090 24G，`probe_freechunker_window3.py`）：

    N=2000 → 2.09 GiB | N=4000 → 4.71 GiB | N=6000 → 9.02 GiB
    N=8000 → 15.03 GiB | N=12000 → OOM

code-repo 域句子数中位 3.7 万+ → 单卡必然爆，且实测报错
`Tried to allocate 23.65 GiB`（单次分配就超整卡）。

修法
----
在 `encode()` 里把句子按 `FC_MAX_SENTENCES`（默认 4000）分段编码，
把各段的 embedding 与分组结果顺序拼接。

为什么拼接是安全的（已实测验证）
------------------------------
不变量：`len(embeddings) == len(sentences) + len(grouped_sentences)`
（probe 实测 N=2 → 3 = 2 + 1 ✅）
`query()` 里 `np.dot(vector_store['embeddings'], q)` 与
`grouped_texts[idx]` 共用同一套下标，所以只要每段各自满足该不变量，
顺序拼接后仍满足。

为什么用**全局**索引做 【Begin-n】 标记
------------------------------------
`src/aggregator.py` 靠 `【Begin-x】` 标记**排序并合并重叠片段**。
窗口若用局部索引，跨窗口的片段排序会错乱。
故本补丁保留全局序号 `_st + _j`。

⚠️ 口径偏差：窗口边界处无法形成跨窗口的 chunk。
仅当 N > 窗口大小才有影响；**N ≤ 窗口时与原实现逐位一致**。

幂等：已打过补丁会跳过。
"""
from __future__ import annotations

import re
import sys

P = "src/encoder.py"

src = open(P, encoding="utf-8").read()

if "FC_MAX_SENTENCES" in src:
    print("[patch] 已包含 FC_MAX_SENTENCES，跳过")
    sys.exit(0)

# 匹配 encode() 里「构造 inputs_embeds → 返回」那一段（容忍空白差异）
pat = re.compile(
    r"( *)inputs_embeds = input_embeddings\.unsqueeze\(0\).*?"
    r"return sentences, result_embeddings, grouped_sentences",
    re.S,
)
m = pat.search(src)
if not m:
    print("[patch] ❌ 找不到 encode() 的目标片段，中止（不改文件）")
    sys.exit(1)

ind = m.group(1)          # 原有缩进（12 空格）
new = f"""{ind}_W = int(os.environ.get('FC_MAX_SENTENCES', '4000') or '0')
{ind}if _W <= 0:
{ind}    _W = len(sentences)
{ind}_all_emb, _all_grp = [], []
{ind}for _st in range(0, len(sentences), _W):
{ind}    _seg = input_embeddings[_st:_st + _W]
{ind}    # ⚠️ 必须用**全局**序号：aggregator 靠 【Begin-n】 排序合并
{ind}    _seg_sent = [f"【Begin-{{_st + _j}}】" + _s + f"【End-{{_st + _j}}】"
{ind}                 for _j, _s in enumerate(sentences[_st:_st + _W])]
{ind}    _out = self.model(inputs_embeds=_seg.unsqueeze(0),
{ind}                      granularities=self.granularities)
{ind}    _all_emb.append(_out['embedding'].cpu().numpy())
{ind}    _all_grp.extend(
{ind}        self._group_sentences_by_shift_matrix(_seg_sent,
{ind}                                               _out['shift_matrix']))
{ind}sentences = [f"【Begin-{{num}}】" + s + f"【End-{{num}}】"
{ind}             for num, s in enumerate(sentences)]
{ind}result_embeddings = (_all_emb[0] if len(_all_emb) == 1
{ind}                     else np.concatenate(_all_emb, axis=0))
{ind}grouped_sentences = _all_grp

{ind}return sentences, result_embeddings, grouped_sentences"""

src = src[:m.start()] + new + src[m.end():]
open(P, "w", encoding="utf-8").write(src)

# 语法检查
import ast  # noqa: E402
ast.parse(open(P, encoding="utf-8").read())
print("[patch] ✅ 已给 encode() 加句子窗口，语法检查通过")
print("        窗口大小由环境变量 FC_MAX_SENTENCES 控制（默认 4000）")
