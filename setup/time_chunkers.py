"""量测 PPL / Margin chunker 在短/中/长文档上的单样本耗时。

用途：决定这两个 chunker 全量 503 样本要跑多久，从而规划是否值得并行、
是否需要换更小的模型或改并行策略。
"""
import os
import sys
import time

sys.path.insert(0, "/home1/lh/FreeChunker")

from datasets import load_from_disk  # noqa: E402
from src.paths import LONGBENCH_V2  # noqa: E402

SPLIT = os.path.join(LONGBENCH_V2, "split_by_domain")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

which = sys.argv[1] if len(sys.argv) > 1 else "ppl"


def pick(domain, idx):
    ds = load_from_disk(os.path.join(SPLIT, domain))
    s = ds[idx]
    return s["context"], s.get("length", "?")


if which == "ppl":
    from baseline.ppl_chunking import PPLChunking
    t0 = time.time()
    ch = PPLChunking(model_name_or_path="/home1/lh/FreeChunker/models/Qwen2-1.5B-Instruct", device="cuda")
    print(f"[load] {time.time()-t0:.1f}s", flush=True)
    for dom, idx in [("long-structured", 0), ("single-doc", 0), ("long-icl", 0)]:
        ctx, lg = pick(dom, idx)
        print(f"\n--- {dom}[{idx}] len={lg} chars={len(ctx)} ---", flush=True)
        n = len(ctx)
        t0 = time.time()
        try:
            chunks = ch.chunk(ctx, language="en")
            dt = time.time() - t0
            print(f"    {len(chunks)} chunks in {dt:.1f}s  ({n/max(dt,1e-6):.0f} char/s)", flush=True)
        except Exception as e:
            print(f"    ERROR {type(e).__name__}: {e}", flush=True)

elif which == "margin":
    from baseline.margin_sampling_chunking import MarginSamplingChunking
    t0 = time.time()
    ch = MarginSamplingChunking(model_name_or_path="/home1/lh/FreeChunker/models/Qwen2-1.5B-Instruct")
    print(f"[load] {time.time()-t0:.1f}s", flush=True)
    for dom, idx in [("long-structured", 0), ("single-doc", 0), ("long-icl", 0)]:
        ctx, lg = pick(dom, idx)
        print(f"\n--- {dom}[{idx}] len={lg} chars={len(ctx)} ---", flush=True)
        n = len(ctx)
        t0 = time.time()
        try:
            chunks = ch.chunk(ctx, language="en")
            dt = time.time() - t0
            print(f"    {len(chunks)} chunks in {dt:.1f}s  ({n/max(dt,1e-6):.0f} char/s)", flush=True)
        except Exception as e:
            print(f"    ERROR {type(e).__name__}: {e}", flush=True)

print("TIMING_DONE")
