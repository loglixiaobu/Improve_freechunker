#!/usr/bin/env python3
"""下游评测统一入口（最小闭环：FreeChunker + Traditional，仅 jina 嵌入模型）。

为什么需要这个脚本：仓库里的 6 个 `test_*.py` 各自都写了完整的 main()，
硬编码了
    dataset_path = '../Data/LongBench-v2'
    encoder_model_path='../saved_models/2-epoch/jina-embeddings-v2-small-en/jina_epoch_1'
    sizes = [256, 512, 1024] / embed_models = [jina, bge-m3, nomic]
    repeats = 3
全部跑一遍 ≈ 60h+。而且 `../Data/` 在我们服务器上不存在（数据在 `datasets/`）。

本脚本**不改动仓库原文件**，而是 import 它们里面的类，用我们自己的参数调用：
  - 语义、判分、topk、aggregation 逻辑 100% 复用官方实现（保证是复现不是重写）
  - 只覆盖 数据集路径 / 模型路径 / 域名 / topk / repeats
  - 输出统一 markdown + raw jsonl，便于对照

用法：
  python eval_downstream.py --method freechunker --domains single-doc --limit 10
  python eval_downstream.py --method traditional --domains single-doc,multi-doc
  python eval_downstream.py --method all
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paths import LONGBENCH_V2, FREE_CHUNK_JINA, DATASETS, ROOT, JINA_LOCAL  # noqa: E402

# 论文/仓库默认的 6 个域
ALL_DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-dialogue", "long-icl", "long-structured"]

# Traditional 的切分尺寸（仓库默认 256/512/1024）
TRAD_SIZES = [256]
# 我们只训了 jina，另外两个不跑
#
# ⚠️ 这里必须传**本地目录**而不是 HF 上的 repo id：
# `src/sentenizer.py` 用 SentenceTransformer(model_name) 加载，
# 传 "jinaai/jina-embeddings-v2-small-en" 会去 huggingface.co 拉配置，
# 而服务器上 huggingface.co 不通（要 HF_ENDPOINT 走镜像）。
# 直接用项目内下好的 models/jina-embeddings-v2-small-en 最稳，零网络依赖。
EMBED_MODELS = [JINA_LOCAL]


def _load_module(path: str, name: str):
    """按路径 import 一个模块（这些 test_*.py 不是包的一部分，没有 __init__）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _summarize(domain_stats_runs, topk_values, method, extra=""):
    """把若干次 run 的 domain_stats 聚合成 mean±std 文本块（复用仓库口径）。"""
    buf = io.StringIO()
    domains = list(domain_stats_runs[0].keys())

    def q(d):
        return d.get("total_questions", d.get("total_samples", 0))

    total_questions_runs = [sum(q(d) for d in r.values()) for r in domain_stats_runs]
    total_time_runs = [sum(d["total_time"] for d in r.values()) for r in domain_stats_runs]

    topk_acc_runs = {k: [] for k in topk_values}
    for r in domain_stats_runs:
        for k in topk_values:
            corr = sum(d["topk_stats"][k]["correct"] for d in r.values())
            tot = sum(d["topk_stats"][k]["total"] for d in r.values())
            topk_acc_runs[k].append((corr / tot) if tot > 0 else 0.0)

    mq = statistics.mean(total_questions_runs)
    mt = statistics.mean(total_time_runs)
    st = statistics.stdev(total_time_runs) if len(total_time_runs) > 1 else 0.0
    avg_list = [t / n if n > 0 else 0.0 for t, n in zip(total_time_runs, total_questions_runs)]
    ma = statistics.mean(avg_list)
    sa = statistics.stdev(avg_list) if len(avg_list) > 1 else 0.0

    p = lambda s: print(s, file=buf)  # noqa: E731
    p("")
    p("=" * 64)
    p(f"方法: {method}   嵌入: {EMBED_MODELS[0]}   {extra}")
    p("=" * 64)
    p(f"题目数: {int(mq)}")
    p(f"总耗时: {mt:.2f}s ± {st:.2f}s")
    p(f"单题耗时: {ma:.2f}s ± {sa:.2f}s")
    p("")
    p("总体准确率:")
    for k in topk_values:
        macc = statistics.mean(topk_acc_runs[k])
        sacc = statistics.stdev(topk_acc_runs[k]) if len(topk_acc_runs[k]) > 1 else 0.0
        p(f"  TopK-{k:<3d}: {macc*100:.2f}% ± {sacc*100:.2f}%")
    p("")
    p("分域明细:")
    for domain in domains:
        tts = [r[domain]["total_time"] for r in domain_stats_runs]
        tqs = [q(r[domain]) for r in domain_stats_runs]
        ats = [t / n if n > 0 else 0.0 for t, n in zip(tts, tqs)]
        line = f"  {domain:16s} n={int(statistics.mean(tqs)):<4d} time={statistics.mean(tts):7.1f}s"
        for k in topk_values:
            accs = []
            for r in domain_stats_runs:
                c = r[domain]["topk_stats"][k]["correct"]
                t = r[domain]["topk_stats"][k]["total"]
                accs.append((c / t) if t > 0 else 0.0)
            line += f"  TopK-{k}={statistics.mean(accs)*100:5.1f}%"
        p(line)
    return buf.getvalue()


def _apply_keep(qa, domains, limit, offset=0, shard=0, nshards=1):
    """把 qa.datasets 裁剪成要跑的域 + 分片切片。

    分片规则：先在「域」这一层切。`nshards` 个 worker 按域名轮流领任务
    （worker k 领 [k::nshards]），这样每个 worker 只加载自己那几个域的数据，
    显存/内存都不会撞。

    ⚠️ 数据来源必须用 qa 自己加载的那份：
      * FreeChunker 走 `split_by_domain`（原始 LongBench，靠 encoder 现场分块）
      * 5 个 baseline 走各自的 chunked 目录（带 `chunks` 字段）
    早期版本这里写死了 `split_by_domain`，导致 baseline 拿到没有 `chunks` 的
    原始样本 → `KeyError: 'chunks'`。
    """
    from datasets import load_from_disk

    picked = domains[shard::nshards] if nshards > 1 else domains

    # 1) 如果 QASystem 已经把数据集加载成 DatasetDict，直接用它的
    src = getattr(qa, "datasets", None)
    if isinstance(src, dict) or hasattr(src, "keys"):
        keep = {}
        for dn in picked:
            if dn not in src:
                continue
            ds = src[dn]
            if offset or limit:
                hi = len(ds) if not limit else min(len(ds), offset + limit)
                ds = ds.select(range(offset, hi))
            keep[dn] = ds
        if keep:
            return keep
        print("[keep] qa.datasets 里没有匹配的域，回退到 split_by_domain")

    # 2) 兜底：从 split_by_domain 读（FreeChunker 的情况）
    keep = {}
    for dn in picked:
        ds = load_from_disk(os.path.join(LONGBENCH_V2, "split_by_domain", dn))
        if offset or limit:
            hi = len(ds) if not limit else min(len(ds), offset + limit)
            ds = ds.select(range(offset, hi))
        keep[dn] = ds
    return keep


# 每个方法：显示名 / 测试模块 / 数据集根目录 / QASystem 的构造参数名
METHODS = {
    "freechunker": {
        "label": "FreeChunker",
        "module": "test_freechunker.py",
        "cls": "EncoderQASystem",
        "data_root": None,  # 直接用 split_by_domain
        "data_kw": None,
        "extra_kw": {"encoder_model_name": "jina", "encoder_model_path": FREE_CHUNK_JINA,
                     "do_sample": False, "temperature": 0.0},
    },
    "traditional": {
        "label": "Traditional(size={size})",
        "module": "test_Traditional.py",
        "cls": "QASystem",
        "data_root": os.path.join(ROOT, "LongBench-v2_chunked", "Traditional", "{size}"),
        "data_kw": "traditional_dataset_path",
        "extra_kw": {},
        "sizes": [256],
    },
    "ppl": {
        "label": "PPL",
        "module": "test_ppl.py",
        "cls": "QASystem",
        "data_root": os.path.join(ROOT, "LongBench-v2_chunked", "PPL", "Qwen2.5-1.5B-Instruct"),
        "data_kw": "ppl_dataset_path",
        "extra_kw": {},
    },
    "margin": {
        "label": "Margin",
        "module": "test_margin.py",
        "cls": "QASystem",
        "data_root": os.path.join(ROOT, "LongBench-v2_chunked", "Margin", "Qwen2.5-1.5B-Instruct"),
        "data_kw": "margin_dataset_path",
        "extra_kw": {},
    },
    "semantic": {
        "label": "Semantic",
        "module": "test_semantic.py",
        "cls": "QASystem",
        "data_root": os.path.join(ROOT, "LongBench-v2_chunked", "Semantic", "bge-m3"),
        "data_kw": "semantic_dataset_path",
        "extra_kw": {},
    },
    "lumber": {
        "label": "Lumber",
        "module": "test_lumber.py",
        "cls": "QASystem",
        "data_root": os.path.join(ROOT, "LongBench-v2_chunked", "Lumber", "qwen3-8b"),
        "data_kw": "lumber_dataset_path",
        "extra_kw": {},
    },
}


def run_method(method, domains, topk_values, repeats, limit, out_dir,
               offset=0, shard=0, nshards=1, chunk_sizes=None, vllm_port=None):
    """跑一个方法。所有方法共用同一套流程，只是 QASystem 类和数据路径不同。"""
    cfg = METHODS[method]
    mod = _load_module(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["module"]),
                       f"fc_test_{method}_{shard}")

    # 让 VLLMClient 走指定端口（多实例时按 worker 分摊）
    if vllm_port:
        _patch_vllm_port(mod, vllm_port)

    sizes = chunk_sizes if chunk_sizes else cfg.get("sizes", [None])
    blocks = []
    for size in sizes:
        # 数据集路径
        if cfg["data_root"] is None:
            data_path = None  # freechunker 走 split_by_domain
        else:
            data_path = cfg["data_root"].format(size=size) if size is not None else cfg["data_root"]
            if not os.path.isdir(data_path):
                raise FileNotFoundError(
                    f"[{method}] 缺少分块数据：{data_path}\n"
                    f"请先跑对应的 chunk 脚本。"
                )

        stats_runs = []
        for rep in range(repeats):
            kw = dict(cfg["extra_kw"])
            if data_path is not None:
                kw[cfg["data_kw"]] = data_path
            if method == "freechunker":
                kw["dataset_path"] = os.path.join(LONGBENCH_V2, "split_by_domain")

            # fc-patch: 所有 QASystem 的嵌入模型一律指到项目内本地目录，
            # 否则会退回 hub repo-id（'jinaai/jina-embeddings-v2-small-en'），
            # 在离线的服务器上会挂死在 HTTP 重试里（实测 130 线程卡住不退出）。
            _cls = getattr(mod, cfg["cls"])
            try:
                import inspect as _inspect
                _params = _inspect.signature(_cls.__init__).parameters
                if "embedding_model_path" in _params:
                    kw["embedding_model_path"] = EMBED_MODELS[0]
            except (TypeError, ValueError):
                pass

            qa = _cls(**kw)

            keep = _apply_keep(qa, domains, limit, offset, shard, nshards)
            if not keep:
                print(f"[{method}] shard {shard}/{nshards} 没领到域，跳过")
                return ""
            qa.datasets = keep
            print(f"[{method}{'' if size is None else ':'+str(size)}] "
                  f"shard {shard}/{nshards} rep {rep+1}/{repeats} "
                  f"n={ {k: len(v) for k, v in keep.items()} } vllm={vllm_port}")

            results, stats = qa.evaluate_single_run(topk_values=topk_values,
                                                    verbose=False, save_results=False)
            stats_runs.append(stats)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = f"{method}" + ("" if size is None else f"_{size}")
            raw = os.path.join(out_dir, f"raw_{tag}_shard{shard}_rep{rep}_{ts}.jsonl")
            with open(raw, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{method}] raw -> {raw} ({len(results)} 行)")

        label = cfg["label"].format(size=size) if size is not None else cfg["label"]
        if nshards > 1:
            label += f" shard{shard}"
        blocks.append(_summarize(stats_runs, topk_values, label,
                                 extra=f"repeats={repeats} shard={shard}/{nshards}"))
    return "\n".join(blocks)


def _patch_vllm_port(mod, port):
    """把模块里所有 localhost:8888 换成指定端口（多 vLLM 实例分摊负载）。"""
    import re
    base = f"localhost:{port}"
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, str) and "localhost:8888" in obj:
            setattr(mod, name, obj.replace("localhost:8888", base))
    # 闭包/方法内部的字面量改不了，用环境变量兜底 + 猴子补丁 requests.get
    os.environ["FC_VLLM_BASE"] = f"http://{base}"
    try:
        import requests as _rq
        _orig_get = _rq.get

        def _get(url, *a, **k):
            if isinstance(url, str) and "localhost:8888" in url:
                url = url.replace("localhost:8888", base)
            return _orig_get(url, *a, **k)

        _rq.get = _get
    except Exception:
        pass
    try:
        from openai import OpenAI as _OI
        _orig_init = _OI.__init__

        def _init(self, *a, **k):
            if isinstance(k.get("base_url"), str) and "localhost:8888" in k["base_url"]:
                k["base_url"] = k["base_url"].replace("localhost:8888", base)
            return _orig_init(self, *a, **k)

        _OI.__init__ = _init
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="freechunker",
                    help="freechunker / traditional / ppl / margin / semantic / lumber / all")
    ap.add_argument("--domains", default="single-doc",
                    help="逗号分隔，或 all。默认 single-doc（最小闭环）")
    ap.add_argument("--topk", default="5,10", help="逗号分隔")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="每域限题数，0=不限（smoke 用）")
    ap.add_argument("--offset", type=int, default=0, help="从第几题开始（分片用）")
    ap.add_argument("--shard", type=int, default=0, help="本 worker 编号")
    ap.add_argument("--nshards", type=int, default=1, help="worker 总数")
    ap.add_argument("--chunk-sizes", default="256", help="Traditional 的 chunk_size 列表")
    ap.add_argument("--vllm-port", type=int, default=None, help="本 worker 用哪个 vLLM 端口")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    domains = ALL_DOMAINS if args.domains == "all" else [d.strip() for d in args.domains.split(",") if d.strip()]
    topk_values = [int(x) for x in args.topk.split(",") if x.strip()]
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]

    out_dir = args.out_dir or os.path.join(ROOT, "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    methods = list(METHODS) if args.method == "all" else [args.method]
    print(f"[eval] methods={methods} domains={domains} topk={topk_values} "
          f"repeats={args.repeats} limit={args.limit} shard={args.shard}/{args.nshards} "
          f"vllm_port={args.vllm_port}")

    blocks = []
    t0 = time.time()
    for m in methods:
        try:
            blocks.append(run_method(m, domains, topk_values, args.repeats, args.limit,
                                     out_dir, args.offset, args.shard, args.nshards,
                                     chunk_sizes if m == "traditional" else None,
                                     args.vllm_port))
        except Exception as e:
            import traceback
            traceback.print_exc()
            blocks.append(f"\n[{m}] 失败: {type(e).__name__}: {e}\n")
    total = time.time() - t0

    header = (f"# 下游评测结果（{ts}）\n\n"
              f"- 方法: {','.join(methods)}\n"
              f"- 域: {','.join(domains)}\n"
              f"- TopK: {topk_values}\n"
              f"- repeats: {args.repeats}, limit: {args.limit}\n"
              f"- shard: {args.shard}/{args.nshards}\n"
              f"- 嵌入模型: {EMBED_MODELS[0]}（仅 jina，另两个未训练）\n"
              f"- QA 生成: Qwen3-8B @ vLLM localhost:{args.vllm_port or 8888}\n"
              f"- 总耗时: {total/60:.1f} min\n")
    body = "\n".join(blocks)
    md = os.path.join(out_dir, f"eval_summary_{args.method}_shard{args.shard}_{ts}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(header + "\n```\n" + body + "\n```\n")
    print(body)
    print(f"\n[eval] 汇总 -> {md}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
