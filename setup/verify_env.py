#!/usr/bin/env python3
"""FreeChunker 复现：环境 / 数据集 / 模型 就绪自检。

用新建的 freechunker 环境运行：
  /home1/lh/miniconda/envs/freechunker/bin/python logs/verify_env.py
"""
import json
import os
import sys
import traceback

ROOT = "/home1/lh/FreeChunker"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"

PASS, FAIL, WARN = [], [], []


def check(name, fn):
    try:
        msg = fn()
        print(f"  [OK]   {name}: {msg}")
        PASS.append(name)
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        FAIL.append(name)


def du(path):
    t = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                t += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return t


def has(path, *names):
    missing = [n for n in names if not os.path.exists(os.path.join(path, n))]
    if missing:
        raise FileNotFoundError(f"缺少 {missing} in {path}")
    return " ".join(names)


print("=" * 74)
print("A. 解释器与 GPU")
print("=" * 74)


def a1():
    import torch
    assert torch.cuda.is_available(), "CUDA 不可用"
    n = torch.cuda.device_count()
    assert n == 8, f"期望 8 张卡，实际 {n}"
    names = {torch.cuda.get_device_name(i) for i in range(n)}
    return f"torch {torch.__version__} | {n} GPUs | {names} | sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}"


check("torch + 8×GPU", a1)


def a2():
    import torch
    a = torch.randn(1024, 1024, device="cuda:0")
    b = (a @ a).sum().item()
    return f"cuda:0 实算 matmul 正常 (sum={b:.1f})"


check("GPU 实算", a2)


def a3():
    import importlib.metadata as md
    want = {
        "torch": "2.9.1", "transformers": "4.57.1", "sentence-transformers": "3.2.0",
        "datasets": "4.4.1", "vllm": "0.16.0", "openai": "2.24.0",
        "huggingface-hub": "0.36.0", "numpy": "2.2.6", "einops": "0.8.1",
        "nltk": "3.9.1", "scikit-learn": "1.7.2", "matplotlib": "3.10.7",
        "pyarrow": "22.0.0", "sentencepiece": "0.2.1", "json_repair": None,
    }
    got, bad = {}, []
    for k, v in want.items():
        try:
            g = md.version(k)
            got[k] = g
            if v and g != v:
                bad.append(f"{k}:{g}!={v}")
        except Exception:
            bad.append(f"{k}:MISSING")
    if bad:
        WARN.append("版本偏差 " + ",".join(bad))
    return f"{len(got)} 个包齐备" + (f"；偏差: {bad}" if bad else "，版本全对齐 dcs")


check("依赖清单", a3)

print()
print("=" * 74)
print("B. 模型（都在项目目录内）")
print("=" * 74)

M = os.path.join(ROOT, "models")


def b1():
    d = os.path.join(M, "jina-embeddings-v2-small-en")
    has(d, "config.json")
    sz = du(d)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(d, trust_remote_code=True, device="cuda:0")
    dim = m.get_sentence_embedding_dimension()
    maxlen = getattr(m, "max_seq_length", "-")
    v = m.encode(["hello world", "another sentence"], show_progress_bar=False)
    assert dim == 512, f"dim={dim} != 512"
    assert v.shape == (2, 512), v.shape
    del m
    import torch
    torch.cuda.empty_cache()
    return f"{sz/1e6:.0f}MB, dim={dim}, max_seq_len={maxlen}, encode OK"


check("jina-embeddings-v2-small-en", b1)


def b2():
    d = os.path.join(M, "FreeChunk-jina")
    has(d, "config.json", "model.safetensors")
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    return (f"{du(d)/1e6:.0f}MB | hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']} "
            f"heads={cfg['num_attention_heads']} vocab={cfg['vocab_size']} "
            f"max_power={cfg.get('max_power')} | auto_map={list(cfg.get('auto_map', {}).keys())}")


check("XiaSheng/FreeChunk-jina (官方权重)", b2)


def b3():
    """用仓库 src 代码加载官方权重，确认跨粒度编码器能实例化并前向"""
    sys.path.insert(0, ROOT)
    import torch
    from src.freechunker import FreeChunkerModel
    d = os.path.join(M, "FreeChunk-jina")
    m = FreeChunkerModel.from_pretrained(d)
    n = sum(p.numel() for p in m.parameters())
    m = m.to("cuda:0").eval()
    E = torch.randn(1, 20, 512, device="cuda:0")
    with torch.no_grad():
        out = m(inputs_embeds=E)
    emb, sm = out["embedding"], out["shift_matrix"]
    assert emb.shape[1] == 512, emb.shape
    del m
    torch.cuda.empty_cache()
    return f"参数量 {n/1e6:.1f}M | 前向 OK: embedding{tuple(emb.shape)} shift_matrix{tuple(sm.shape)}"


check("FreeChunkerModel 加载+前向 (官方权重)", b3)


def b4():
    d = os.path.join(M, "bge-m3")
    has(d, "config.json")
    sz = du(d)
    sft = [f for f in os.listdir(d) if f.endswith(".safetensors") or f.endswith(".bin")]
    return f"{sz/1e9:.2f}GB, 权重文件={sft}"


check("BAAI/bge-m3 (训练初始化用)", b4)


def b5():
    d = os.path.join(M, "Qwen3-8B")
    sz = du(d)
    assert sz > 10e9, f"只有 {sz/1e9:.2f}GB，可能没下完"
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    idx = os.path.join(d, "model.safetensors.index.json")
    nshard = len(json.load(open(idx, encoding="utf-8"))["weight_map"]) if os.path.exists(idx) else "-"
    n = sum(1 for f in os.listdir(d) if f.endswith(".safetensors"))
    return (f"{sz/1e9:.2f}GB | {n} 个 shard | layers={cfg['num_hidden_layers']} "
            f"hidden={cfg['hidden_size']} | max_pos={cfg.get('max_position_embeddings')}")


check("Qwen/Qwen3-8B", b5)

print()
print("=" * 74)
print("C. 数据集（都在项目目录内）")
print("=" * 74)

D = os.path.join(ROOT, "datasets")


def _corpus():
    """HF 上这个仓库没有根 dataset_dict.json，所以 load_from_disk(根目录) 会失败；
    必须按 split 子目录逐个 load_from_disk。load_dataset 同样不可用。"""
    from datasets import load_from_disk
    p = os.path.join(D, "FreeChunk-corpus")
    return p, {"train": load_from_disk(os.path.join(p, "train")),
               "val": load_from_disk(os.path.join(p, "val"))}


def c1():
    p, dd = _corpus()
    tr, va = dd["train"], dd["val"]
    lens = [len(s) for s in tr.select(range(min(2000, len(tr))))["sentences"]]
    import statistics
    tok = sorted({t for t in tr.select(range(min(500, len(tr))))["original_token_count"]})
    return (f"train={len(tr)} val={len(va)} | 每篇句数 mean={statistics.mean(lens):.1f} "
            f"p50={statistics.median(lens):.0f} max={max(lens)} | token 取值={tok}")


check("XiaSheng/FreeChunk-corpus", c1)


def c1b():
    """记录两种常规加载方式在本地布局上都会失败，供改脚本参考"""
    p = os.path.join(D, "FreeChunk-corpus")
    from datasets import load_dataset, load_from_disk
    res = []
    try:
        load_dataset(p, split="train")
        res.append("load_dataset=OK")
    except Exception as e:
        res.append(f"load_dataset=FAIL({type(e).__name__})")
    try:
        load_from_disk(p)
        res.append("load_from_disk(根)=OK")
    except Exception as e:
        res.append(f"load_from_disk(根)=FAIL({type(e).__name__})")
    res.append("load_from_disk(根/train)=OK")
    return " | ".join(res) + "  -> 脚本必须按 split 子目录加载"


check("语料加载方式探测", c1b)


def c2():
    from src.utils import generate_shifted_matrix
    p, dd = _corpus()
    tr = dd["train"]
    n = len(tr[0]["sentences"])

    sm = generate_shifted_matrix(n)[0]
    m_code = int(sm.shape[1])

    # 独立复算，不依赖仓库代码
    m_ref = 0
    for g in (2, 4):
        if g > n:
            continue
        step = max(1, g // 2)
        ms = n - g
        m_ref += len(range(0, ms + 1, step))
        if ms >= 0 and (ms % step) != 0:
            m_ref += 1
    assert m_code == m_ref, f"{m_code} != {m_ref}"

    # 全量统计 span / 向量总数
    import statistics
    sample = tr.select(range(min(2000, len(tr))))
    ns = [len(s) for s in sample["sentences"]]
    ms = [int(generate_shifted_matrix(x)[0].shape[1]) for x in ns]
    tot = statistics.mean([a + b for a, b in zip(ns, ms)])
    return (f"n={n} 句 -> [2,4] span={m_code} 条（代码与独立复算一致）| "
            f"2000 篇均值: 句 {statistics.mean(ns):.1f} + span {statistics.mean(ms):.1f} = "
            f"{tot:.1f} 条/篇 -> 100k 篇约 {tot*1e5/1e6:.1f}M 条")


check("granularity [2,4] 口径核对", c2)


def c3():
    p = os.path.join(D, "LongBench-v2", "data.json")
    assert os.path.exists(p), f"找不到 {p}"
    data = json.load(open(p, encoding="utf-8"))
    from collections import Counter
    dom = Counter(d["domain"] for d in data)
    keys = sorted(data[0].keys())
    assert len(data) == 503, f"题数 {len(data)} != 503"
    return f"{len(data)} 题 | {len(dom)} 个 domain | 字段={keys}\n         " + \
        "\n         ".join(f"{k}: {v}" for k, v in sorted(dom.items()))


check("zai-org/LongBench-v2 (data.json)", c3)


def c4():
    """官方训练曲线 —— 后面 G2 验收的靶子"""
    p = os.path.join(M, "FreeChunk-jina", "training_losses.json")
    assert os.path.exists(p), "官方 checkpoint 里没有 training_losses.json"
    d = json.load(open(p, encoding="utf-8"))
    sl = d["training_losses"]["step_losses"]
    vl = d["validation_losses"]["val_losses"]
    import statistics
    return (f"total_steps={d['training_config']['total_steps']} | "
            f"前1000步均值={statistics.mean(sl[:1000]):.4f} | "
            f"末1000步均值={statistics.mean(sl[-1000:]):.4f} | "
            f"final_val={vl[-1]:.4f} (G2 靶子)")


check("官方训练曲线 (G2 验收靶子)", c4)

print()
print("=" * 74)
print("D. 待修代码问题（只报不改，等你确认）")
print("=" * 74)
for desc, path in [
    ("默认 tokenizer 硬编码了不存在的路径 /share/home/ecnuzwx/...", "baseline/traditional_chunking.py"),
    ("load_dataset / load_from_disk(根) 在本地下好的语料上都会失败，需按 split 子目录加载",
     "train and preprocess/1_build_pretrain_datasets.py"),
    ("造数据脚本存到 ./vector/jina-.../part_N，训练脚本读 ./vector/jina-.../train",
     "1_build_pretrain_datasets.py + 3_train_jina.py"),
    ("src/ 无 __init__.py，需 PYTHONPATH=. 运行", "src/"),
    ("测试脚本 dataset_path 为相对路径 ../Data/LongBench-v2；model path 也与训练脚本输出不一致", "test/test_freechunker.py"),
]:
    print(f"  [待修] {desc}\n         -> {path}")

print()
print("=" * 74)
print(f"结果：PASS={len(PASS)}  FAIL={len(FAIL)}  WARN={len(WARN)}")
if FAIL:
    print("未通过：", FAIL)
if WARN:
    print("提示：", WARN)
print("VERIFY_ALL_OK" if not FAIL else "VERIFY_HAS_FAILURES")
print("=" * 74)
