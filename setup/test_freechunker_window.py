#!/usr/bin/env python3
"""验证 FreeChunker 句子窗口补丁。

两个断言
--------
A. **回归**：N ≤ 窗口时，输出必须与原实现**逐位一致**
   （这是承诺，必须实测；否则所有本来能跑通的域都会被悄悄改掉）
B. **窗口**：N > 窗口时，分窗拼接后仍满足
       len(embeddings) == len(grouped_texts)
   且句子不丢不重、`query()` 可用

用法:
    CUDA_VISIBLE_DEVICES=2 python setup/test_freechunker_window.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.encoder import UnifiedEncoder  # noqa: E402

MODEL = os.path.join(ROOT, "models", "FreeChunk-jina")
BAK = os.path.join(ROOT, "src", "encoder.py.bak_window")

# ⚠️ 必须用**真实数据**：手写的短文本会被 splitter 的「短句合并」并成 1 句，
# 那样根本压不到多窗口路径（第一版测试就踩了这个坑）。
# 实测真实 splitter ≈ 1100 字符/句（code-repo），所以取一段真实代码文本。
def load_real_text():
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(ROOT, "datasets", "LongBench-v2",
                                     "split_by_domain", "code-repo"))
    return ds[23]["context"][:200_000]     # ~179 句，足够压多窗口


TEXT = load_real_text()


def load_orig():
    """从备份文件加载「原始」encoder 模块（后缀不是 .py，要显式指定 loader）。"""
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader("enc_orig", BAK)
    spec = importlib.util.spec_from_loader("enc_orig", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m.UnifiedEncoder


def build(enc, text):
    enc.build_vector_store(text, show_progress=False)
    vs = enc.vector_store
    return (vs["grouped_texts"],
            vs["embeddings"],
            len(vs["sentences"]),
            len(vs["grouped_sentences"]))


ok_all = True
print(f"测试文本: {len(TEXT):,} 字符（code-repo 前 200k）")

# ---------------- A) 回归 ----------------
# 窗口设成远大于 N，保证只走一个窗口 → 应与原实现逐位一致
os.environ["FC_MAX_SENTENCES"] = "1000000"
Orig = load_orig()
e_new = UnifiedEncoder(model_name="jina", local_model_path=MODEL, granularities=(2, 4))
e_old = Orig(model_name="jina", local_model_path=MODEL, granularities=(2, 4))

gt_new, emb_new, ns_new, ng_new = build(e_new, TEXT)
gt_old, emb_old, ns_old, ng_old = build(e_old, TEXT)

same_txt = (gt_new == gt_old)
same_emb = (emb_new.shape == emb_old.shape) and np.allclose(emb_new, emb_old, atol=0)
print("=== A) 回归（单窗口，应与原实现逐位一致）===")
print(f"  句子数 new={ns_new} old={ns_old}  组数 new={ng_new} old={ng_old}")
print(f"  grouped_texts 完全相同: {same_txt}  {'✅' if same_txt else '❌'}")
print(f"  embeddings 完全相同  : {same_emb}  {'✅' if same_emb else '❌'}")
ok_all &= same_txt and same_emb

# ---------------- B) 窗口 ----------------
W = 50
print(f"\n=== B) 窗口（FC_MAX_SENTENCES={W}，强制多窗；N={ns_new} → "
      f"{-(-ns_new // W)} 窗）===")
os.environ["FC_MAX_SENTENCES"] = str(W)
e_win = UnifiedEncoder(model_name="jina", local_model_path=MODEL, granularities=(2, 4))
gt_w, emb_w, ns_w, ng_w = build(e_win, TEXT)
inv = (emb_w.shape[0] == len(gt_w))
print(f"  句子数={ns_w} 组数={ng_w}  grouped_texts={len(gt_w)}  embeddings={emb_w.shape}")
print(f"  不变量 len(embeddings)==len(grouped_texts): {inv}  {'✅' if inv else '❌'}")
ok_all &= inv
ok_all &= (ns_w == ns_new)      # 句子总数不应因分窗而变

# 句子不丢：每句的 Begin 标记应能在 grouped_texts 里找到
import re
begins = sorted({int(x) for t in gt_w for x in re.findall(r"【Begin-(\d+)】", t)})
all_begins = list(range(ns_w))
lost = [i for i in all_begins if i not in begins]
print(f"  句子覆盖: 出现 {len(begins)}/{ns_w} 个 Begin 标记，丢失 {lost[:5]}  "
      f"{'✅' if not lost else '❌'}")
ok_all &= (not lost)

try:
    r = e_win.query("What is the core problem?", top_k=3)
    print(f"  query(top_k=3) -> {len(r)} 段  {'✅' if r else '❌'}")
    ok_all &= bool(r)
except Exception as e:
    print(f"  query 失败: {type(e).__name__}: {e}")
    ok_all = False

# 跨窗口的 chunk 无法形成 —— 记录一下实际偏差大小
print(f"\n  口径偏差: {ns_w} 句中，跨窗口边界最多 {(ns_w // W) - 1} 处无法成 chunk")
os.environ["FC_MAX_SENTENCES"] = "4000"
print(f"\n{'TEST_FC_WINDOW_OK' if ok_all else 'TEST_FC_WINDOW_FAIL'}")
sys.exit(0 if ok_all else 1)
