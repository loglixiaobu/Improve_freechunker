#!/usr/bin/env python3
"""生成「本次复现 vs 论文」的差距表。

数据来源
--------
* 本次实测：`test_result/downstream_all_table.csv`
* 论文数值：FreeChunker.pdf 的 Table 1（jina-embeddings-v2-small-en 行），
  逐列核对过：列顺序 = Traditional / SemanticChunker / PPL / Margin /
  Lumber / FreeChunker，每方法两列 = Top-5 / Top-10。

用法:
    python setup/make_paper_gap.py
输出:
    test_result/paper_gap.md
"""
from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "test_result")

# ---- 论文 Table 1（jina-embeddings-v2-small-en）----
# 域顺序与论文一致：I..VI
PAPER = {
    "Traditional": {
        "single-doc": (30.86, 35.43), "multi-doc": (24.80, 28.00),
        "code-repo": (44.00, 51.33), "long-icl": (27.57, 30.86),
        "long-dialogue": (26.50, 33.33), "long-structured": (15.15, 24.24),
        "Overall": (28.76, 33.53),
    },
    "FreeChunker": {
        "single-doc": (35.43, 32.57), "multi-doc": (26.40, 26.40),
        "code-repo": (46.00, 44.00), "long-icl": (32.10, 28.40),
        "long-dialogue": (28.21, 23.08), "long-structured": (42.42, 45.45),
        "Overall": (33.60, 31.61),
    },
}
# 论文域标题顺序（用于展示）
PAPER_ORDER = ["single-doc", "multi-doc", "code-repo",
               "long-icl", "long-dialogue", "long-structured"]
PAPER_TITLE = {
    "single-doc": "I. Single-Document QA",
    "multi-doc": "II. Multi-Document QA",
    "code-repo": "III. Code Repository",
    "long-icl": "IV. Long In-context Learning",
    "long-dialogue": "V. Long-dialogue History",
    "long-structured": "VI. Long Structured Data",
}

# 本次用哪个方法对应论文的 "Traditional"（论文正文说是 256 tokens）
OUR_TRAD = "Traditional(256)"


def load_ours():
    """-> {(method, domain, topk): (acc, correct, total)}"""
    out = {}
    p = os.path.join(RES, "downstream_all_table.csv")
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["method"], r["domain"], int(r["topk"]))] = (
                float(r["acc_pct"]), int(r["correct"]), int(r["total"]))
    return out


def main():
    ours = load_ours()
    L = []
    L.append("# 本次复现 vs 论文 · 差距表\n")
    L.append("**嵌入模型**：jina-embeddings-v2-small-en　"
             "**QA 生成**：Qwen3-8B @ vLLM　**基准**：LongBench V2\n")
    L.append("论文数值取自 FreeChunker.pdf 的 Table 1"
             "（jina-small-en 行）。差值 = 本次 − 论文，单位百分点。\n")

    for meth, ours_meth in (("FreeChunker", "FreeChunker"),
                            ("Traditional", OUR_TRAD)):
        L.append(f"## {meth}\n")
        L.append(f"（本次对应方法：`{ours_meth}`）\n")
        L.append("| 域 | 论文 Top-5 | 本次 Top-5 | Δ | 论文 Top-10 | 本次 Top-10 | Δ |")
        L.append("|---|---|---|---|---|---|---|")
        for dn in PAPER_ORDER:
            p5, p10 = PAPER[meth][dn]
            o5 = ours[(ours_meth, dn, 5)][0]
            o10 = ours[(ours_meth, dn, 10)][0]
            L.append(f"| {PAPER_TITLE[dn]} | {p5:.2f} | {o5:.2f} | "
                     f"{o5-p5:+.2f} | {p10:.2f} | {o10:.2f} | {o10-p10:+.2f} |")
        p5, p10 = PAPER[meth]["Overall"]
        # 本次 Overall 用加权（题数加权），与论文口径一致
        tot5 = sum(ours[(ours_meth, d, 5)][2] for d in PAPER_ORDER)
        cor5 = sum(ours[(ours_meth, d, 5)][1] for d in PAPER_ORDER)
        tot10 = sum(ours[(ours_meth, d, 10)][2] for d in PAPER_ORDER)
        cor10 = sum(ours[(ours_meth, d, 10)][1] for d in PAPER_ORDER)
        o5, o10 = 100.0 * cor5 / tot5, 100.0 * cor10 / tot10
        L.append(f"| **Overall** | **{p5:.2f}** | **{o5:.2f}** | "
                 f"**{o5-p5:+.2f}** | **{p10:.2f}** | **{o10:.2f}** | "
                 f"**{o10-p10:+.2f}** |")
        L.append("")

    L.append("## 结论\n")
    L.append("1. **Traditional 几乎完全复现**：6 个域里有 2 个**逐位相同**"
             "（single-doc 30.86/35.43、long-structured 15.15/24.24），"
             "其余域的 Top-10 也基本吻合。")
    L.append("   → 说明**检索 + QA + 判分整条链路是忠实的**，"
             "FreeChunker 的差距不是评测流程造成的。\n")
    L.append("2. **FreeChunker 有系统性偏差**，集中在长上下文域"
             "（long-structured、long-icl）。\n")
    L.append("3. **已排除：granularity 配置不一致（我一开始的猜测是错的）。**")
    L.append("   论文正文写 *\"the simplest combination of three granularities is "
             "directly preset: **(1, 2, 4)**\"*，")
    L.append("   而代码里 `generate_shifted_matrix` 的默认值是 `[2, 4]`，看着像少了一档。")
    L.append("   但 `src/encoder.py` 里：")
    L.append("   ```python")
    L.append("   embedding = torch.cat([inputs_embeds, sequence_output], dim=1)")
    L.append("   ```")
    L.append("   `inputs_embeds` 就是 **Sentenizer 产出的原始句子向量 = granularity 1**，")
    L.append("   与模型产出的 g=2/g=4 chunk 向量拼接后一起进向量库。")
    L.append("   实测 N=179 句时 `grouped_texts = 179 + 267 = 446`"
             "（179 个单句 + 267 个多句 chunk）→ **配置本就是 (1,2,4)**。\n")
    L.append("4. **更可能的候选：我为绕开 OOM 加的「句子窗口」补丁。**")
    L.append("   `src/encoder.py` 现按 `FC_MAX_SENTENCES=4000` 分段编码。")
    L.append("   已验证 **N ≤ 4000 句时与原实现逐位一致**，")
    L.append("   但 N > 4000 的长文档会在窗口边界处**无法形成跨窗口 chunk**。")
    L.append("   而差距最大的 long-structured / long-icl 正是上下文最长的两个域 ——")
    L.append("   **这条更值得先查**：统计各域有多少样本的句子数超过 4000。\n")
    L.append("5. 其他待排查项：vLLM 上下文上限 32768（原代码 40000）"
             "对长上下文域的系统性影响。\n")
    L.append("## 本次未能对照的部分\n")
    L.append("论文 Table 1 还有 SemanticChunker / PPL / Margin / Lumber 四个方法，"
             "本次因 GPU 硬件故障未能跑完，暂无法对照。\n")

    out = os.path.join(RES, "paper_gap.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
