#!/usr/bin/env python3
"""构建 FreeChunker 训练用的 (input, label) 向量对。

每条记录：
  input = 该文档每个「原子句」用 backbone 编码得到的向量（n 条，每条 512 维）
  label = 该文档每个 chunk pattern 覆盖的句子拼接后用 backbone 编码的向量（m 条）

相对原脚本（mazehart/FreeChunker）的修改：

  1. **语料按 split 子目录 load_from_disk**。本地布局没有根 dataset_dict.json，
     原脚本的 `load_dataset(hub_id, split=...)` 在本地会失败（详见 src/paths.py）。
  2. **新增 --shard/--nshards、--start/--end、--corpus-split**，支持 8 卡并行 +
     精确切片（原脚本的 val 只编 0:200、test 编 200:500，靠 start/end 表达）。
  3. **part 文件名带全局下标**（`part_{block_start:06d}`），分片之间不会撞名；
     按 block 组织，**已存在的 part 自动跳过**，可断点续跑。
  4. **按长度分桶打包 batch**（见 plan_batches）。原脚本写死 batch_size=8，
     而语料里最长句子 3471 token、最长 chunk 7078 token，jina 的注意力是
     B*H*L^2*4 字节 —— 盲目用大 batch 会 OOM，盲目用小 batch 又浪费短句的并行度。
  5. **OOM 二分重试**，保证个别超长文本不会让整片任务挂掉。
  6. **显式落成 float32**（原脚本没 cast，Arrow 从 float32 numpy 推断也是 float32，
     这里显式写死避免哪天上游改了推断规则让磁盘翻倍）。
     input/label 是**二维**列表 [n, 512]，所以是 Sequence(Sequence(float32))。
  7. 每个分片写一份 manifest（lo/hi/篇数/句数/向量数/耗时），供 G1 门禁核对。
  8. 打印吞吐与 ETA。

粒度固定 [2,4]：与 src.utils.generate_shifted_matrix 的默认值、
官方权重 config.json 的 max_power=4 一致（论文正文写的 5 个粒度与代码不符，以代码为准）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from sentence_transformers import SentenceTransformer

from src.paths import GRANULARITIES, JINA_LOCAL, VECTOR_DIR, load_corpus_split
from src.utils import generate_shifted_matrix

# input/label 每条是 [n, 512] 的二维列表 —— 必须是两层 Sequence。
# 只写一层会在 cast 时报：
#   TypeError: Couldn't cast array of type list<item: float> to float
FEATURES = Features({
    "input": Sequence(Sequence(Value("float32"))),
    "label": Sequence(Sequence(Value("float32"))),
})

EMB_DIM = 512
# 注意力显存 ≈ B * H * L^2 * 4 字节。约束 B*L^2 <= SQ_BUDGET 就把它压在 ~2.6 GB。
SQ_BUDGET = 8.0e7
MAX_BS = 32


def build_comb_list(sentences, granularities):
    """按 chunk pattern mask 生成「句子拼接」列表。

    generate_shifted_matrix 返回 [1, n, m]，取 [0] 得 [n, m]，
    第 col 列中为 1 的行就是该 pattern 覆盖的句子下标。
    """
    n = len(sentences)
    matrix = generate_shifted_matrix(n, granularities=list(granularities))[0]
    comb = []
    for col in range(matrix.shape[1]):
        idx = (matrix[:, col] == 1).nonzero(as_tuple=True)[0].tolist()
        if idx:
            comb.append(" ".join(sentences[i] for i in idx))
    return comb, int(matrix.shape[1])


def plan_batches(texts, max_bs=MAX_BS, sq_budget=SQ_BUDGET):
    """按长度打包 batch，约束 `len(batch) * Lmax^2 <= sq_budget`。

    L 用**字符长度**当 token 数的上界（英文约 4 字符/token，中日韩 1~2 字符/token，
    取字符数最保守）。这样：占绝大多数的短文本仍能凑满 max_bs 条（速度不受影响），
    只有长尾会被自动降到 1~2 条。

    返回的是下标列表的列表（按长度升序分组），调用方需自行按原下标回填。
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = max(1, len(texts[i]))
        nm = max(cur_max, L)
        if cur and (len(cur) >= max_bs or (len(cur) + 1) * nm * nm > sq_budget):
            batches.append(cur)
            cur, cur_max = [], 0
            nm = L
        cur.append(i)
        cur_max = nm
    if cur:
        batches.append(cur)
    return batches


def _encode_batch(model, batch):
    v = model.encode(batch, batch_size=len(batch), show_progress_bar=False,
                     convert_to_numpy=True)
    return np.asarray(v, dtype=np.float32)


def encode_batch_safe(model, batch):
    """编码一个 batch；OOM 就二分重试，避免个别超长文本让整片任务崩掉。"""
    try:
        return _encode_batch(model, batch)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()):
            raise
        torch.cuda.empty_cache()
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        a = encode_batch_safe(model, batch[:mid])
        b = encode_batch_safe(model, batch[mid:])
        return np.concatenate([a, b], axis=0)


def encode_all(model, texts, stats):
    """返回与 texts **同序**的向量列表。"""
    if not texts:
        return []
    out = [None] * len(texts)
    for idxs in plan_batches(texts):
        batch = [texts[i] for i in idxs]
        v = encode_batch_safe(model, batch)
        for k, i in enumerate(idxs):
            out[i] = v[k]
        stats["n_batches"] += 1
        stats["max_batch"] = max(stats["max_batch"], len(batch))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    help="输出用的逻辑 split 名（train/val/test/...）")
    ap.add_argument("--corpus-split", default=None,
                    help="从哪个语料子目录读；默认同 --split（test 需要显式指定 val）")
    ap.add_argument("--out-dir", default=None,
                    help="part 输出目录；默认 <VECTOR_DIR>/_parts/<split>")
    ap.add_argument("--shard", type=int, default=0, help="第几个分片（0-based）")
    ap.add_argument("--nshards", type=int, default=1, help="总共分几片")
    ap.add_argument("--start", type=int, default=None, help="显式起始下标（覆盖 --shard）")
    ap.add_argument("--end", type=int, default=None, help="显式结束下标（不含）")
    ap.add_argument("--slice-size", type=int, default=1000, help="每个 part 多少篇")
    ap.add_argument("--max-batch", type=int, default=MAX_BS)
    ap.add_argument("--sq-budget", type=float, default=SQ_BUDGET)
    ap.add_argument("--model", default=JINA_LOCAL)
    ap.add_argument("--granularities", default=",".join(map(str, GRANULARITIES)))
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 篇（冒烟用）")
    ap.add_argument("--force", action="store_true", help="已存在的 part 也重算")
    args = ap.parse_args()

    split = args.split
    corpus_split = args.corpus_split or ("val" if split == "validation" else split)
    granularities = tuple(int(x) for x in args.granularities.split(","))
    out_dir = args.out_dir or os.path.join(VECTOR_DIR, "_parts", split)
    os.makedirs(out_dir, exist_ok=True)

    ds = load_corpus_split(corpus_split)
    total = len(ds)

    if args.start is not None or args.end is not None:
        lo = args.start or 0
        hi = args.end if args.end is not None else total
    else:
        per = (total + args.nshards - 1) // args.nshards
        lo = args.shard * per
        hi = min(lo + per, total)
    hi = min(hi, total)
    if args.limit:
        hi = min(hi, lo + args.limit)

    tag = f"[{split}<-{corpus_split} shard={args.shard}/{args.nshards} {lo}:{hi}]"
    print(f"{tag} 语料 {total} 篇；本片 {hi - lo} 篇；part -> {out_dir}", flush=True)
    print(f"{tag} 粒度 {granularities} | max_batch {args.max_batch} | "
          f"sq_budget {args.sq_budget:.0e} | 模型 {args.model}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{tag} device={device} "
          f"{torch.cuda.get_device_name(0) if device == 'cuda' else ''}", flush=True)
    model = SentenceTransformer(args.model, trust_remote_code=True, device=device)
    model.eval()
    dim = model.get_sentence_embedding_dimension()
    print(f"{tag} embedding dim = {dim} | max_seq_length = {model.max_seq_length}", flush=True)
    assert dim == EMB_DIM, f"jina-embeddings-v2-small-en 应为 {EMB_DIM} 维，实际 {dim}"

    blocks = [(b, min(b + args.slice_size, hi)) for b in range(lo, hi, args.slice_size)]
    t_all = time.time()
    n_done = n_skip_docs = n_short = n_vectors = 0
    stats = {"n_batches": 0, "max_batch": 0}

    for bi, (b_lo, b_hi) in enumerate(blocks, 1):
        part = os.path.join(out_dir, f"part_{b_lo:06d}")
        if os.path.exists(part) and not args.force:
            n_skip_docs += load_from_disk(part).num_rows
            print(f"{tag} block {bi}/{len(blocks)} [{b_lo}:{b_hi}] 已存在，跳过", flush=True)
            continue

        records = []
        t0 = time.time()
        v_blk = 0
        for i in range(b_lo, b_hi):
            sentences = ds[i]["sentences"]
            n = len(sentences)
            if n < 2:
                n_short += 1
                continue
            comb, m = build_comb_list(sentences, granularities)
            inputs = encode_all(model, sentences, stats)
            labels = encode_all(model, comb, stats)
            if len(inputs) != n or len(labels) != m:
                raise RuntimeError(f"doc {i}: 编码条数不对 input={len(inputs)}/{n} label={len(labels)}/{m}")
            records.append({"input": inputs, "label": labels})
            v_blk += n + m
            n_done += 1

        dt = time.time() - t0
        n_vectors += v_blk
        part_ds = Dataset.from_list(records).cast(FEATURES)
        part_ds.save_to_disk(part)
        rate = (b_hi - b_lo) / dt if dt > 0 else 0
        print(f"{tag} block {bi}/{len(blocks)} [{b_lo}:{b_hi}] "
              f"{len(records)} 篇 / {dt:.1f}s ({rate:.1f} 篇/s, "
              f"{v_blk/max(dt,1e-6):.0f} 向量/s) -> {os.path.basename(part)}", flush=True)

    dt_all = time.time() - t_all
    print(f"{tag} 完成：处理 {n_done} 篇 / 跳过已存在 {n_skip_docs} 篇 / n<2 跳过 {n_short} 篇；"
          f"共 {n_vectors} 条向量；耗时 {dt_all/60:.1f} min", flush=True)

    manifest = {
        "split": split, "corpus_split": corpus_split,
        "shard": args.shard, "nshards": args.nshards,
        "lo": lo, "hi": hi, "slice_size": args.slice_size,
        "granularities": list(granularities), "model": args.model, "dim": dim,
        "max_seq_length": int(model.max_seq_length),
        "max_batch": args.max_batch, "sq_budget": args.sq_budget,
        "n_records": n_done + n_skip_docs,
        "n_new": n_done, "n_skipped_parts": n_skip_docs, "n_short_docs": n_short,
        "n_vectors": n_vectors,
        "batch_stats": stats,
        "elapsed_min": round(dt_all / 60, 2),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    mp = os.path.join(out_dir, f"manifest_shard{args.shard:02d}.json")
    json.dump(manifest, open(mp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{tag} manifest -> {mp}", flush=True)
    print(f"{tag} PART_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
