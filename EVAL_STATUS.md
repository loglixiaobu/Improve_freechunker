# 下游评测状态（2026-10-04）

论文 §5 的 baseline 对比。**最小闭环**：先跑 FreeChunker + Traditional 两个方法，
只用一个嵌入模型（jina），只跑 single-doc + multi-doc 两个域。

## 为什么要缩范围

论文原表是 6 方法 × 3 嵌入模型 × 6 域 × 3 repeats。全量跑一轮：
- QA 生成次数 = 503 题 × 6 方法 × 3 模型 × 3 repeats × 2 topk ≈ **54,000 次**
- 单次约 2s → **30h**，加上分块和编码（PPL/Margin/Lumber 每个都要过一遍 LLM 分块）实际 **60h+**

而且我们只训了 jina 一个模型（bge-m3 / nomic 的权重没有），全量本来就跑不全。
所以先做最小闭环：**验证链路通 + 拿到 FreeChunker vs Traditional 的对比数字**。

## 环境

| 组件 | 配置 |
|---|---|
| vLLM | Qwen3-8B，单卡 GPU 0，端口 8888，`max_model_len=16384`，`cuda_visible=0` |
| 评测编码器 | GPU 1（`CUDA_VISIBLE_DEVICES=1`），避免和 vLLM 抢显存 |
| 分块 | Traditional chunk_size=256 token（Qwen3-8B tokenizer，本地权重） |
| 嵌入 | `models/jina-embeddings-v2-small-en`（本地目录，不走 HF） |

**踩坑**：第一版 vLLM 用 `max_model_len=40960` 起不来 —— 模型占 15.27 GiB，
24G 卡只剩 3.32 GiB 给 KV cache，而 40960 需要 5.62 GiB。
vLLM 自己在报错里说了上限 24144，改成 16384 后正常（KV cache 4.97 GiB，并发 2.21x）。

## 阶段进度

| 阶段 | 状态 | 产物 |
|---|---|---|
| 0 vLLM 启动 | ✅ 11:02 就绪 | `logs/vllm.log` |
| 1 LongBench 拆域 | ✅ 503 题 → 6 个域 | `datasets/LongBench-v2/split_by_domain/` |
| 2 Traditional 分块 | ✅ 256 token | `LongBench-v2_chunked/Traditional/256/` |
| 3 FreeChunker 评测 | ✅ 300 题 / 600 次 QA，9.2 min | `eval_results/eval_summary_20261004_111136.md` |
| 4 Traditional 评测 | 🔄 运行中 | `logs/eval_traditional.log` |

## 已有结果：FreeChunker

```
题目数 300（single-doc 175 + multi-doc 125）
总耗时 549s，单题 1.83s

TopK-5  : 30.00%
TopK-10 : 30.00%

single-doc   n=175  TopK-5=33.1%  TopK-10=34.3%
multi-doc    n=125  TopK-5=25.6%  TopK-10=24.0%
```

⚠️ **这批数字要打折扣**：它用的是 `models/FreeChunk-jina`（官方权重），但当时
sentenizer 还在走 huggingface.co（我漏了 `HF_ENDPOINT`），是侥幸加载成功的缓存副本。
现在已改成显式加载本地目录，等 Traditional 跑完会重跑一遍确认数字稳定。

## 怎么查 / 怎么续

```bash
# 看评测会话
tmux attach -t eval        # 或 tmux capture-pane -p -t eval
tail -f ~/FreeChunker/logs/eval_traditional.log

# 续跑（每阶段幂等，分块/评测产物在就跳过）
bash -lc "source ~/miniconda/etc/profile.d/conda.sh && conda activate freechunker \
  && export PYTHONPATH=~/FreeChunker && cd ~/FreeChunker \
  && python setup/run_eval.py --skip-vllm"

# 只重跑某一半
python setup/run_eval.py --skip-vllm --only freechunker
python setup/run_eval.py --skip-vllm --skip-chunk --only traditional
```

## 关键：改动了哪些文件

| 文件 | 改动 |
|---|---|
| `test/eval_downstream.py` | **新增**。统一评测入口，复用官方 `EncoderQASystem` / `QASystem`，只覆盖路径与参数 |
| `setup/run_eval.py` | **新增**。三阶段编排 + vLLM 拉起 + 幂等跳过 |
| `setup/start_vllm.sh` | **新增**。vLLM 启动（单卡、本地权重、16384 上下文） |
| `train and preprocess/9_chunk_traditional.py` | 改：tokenizer 走本地 `models/Qwen3-8B`；数据读 `split_by_domain` 的 `load_from_disk`；补 `sys.path`（原文件备份 `.orig`） |

**没有改动**任何 `test/test_*.py` 和 `baseline/*.py` 的评估逻辑 —— 判分、topk、
聚合口径 100% 是官方实现。这样出来的对比才叫复现。

## 三个已确认的坑

1. **`load_dataset(<split_by_domain 目录>)` 会把 6 个域压成一个 `train` split**，
   域名全丢 → 必须逐域 `load_from_disk`。
2. **`load_dataset(dir, "single-doc")` 报 `BuilderConfig 'single-doc' not found`** ——
   `save_to_disk` 出来的目录没有 config 概念。同上，用 `load_from_disk`。
3. **vLLM 占着 GPU 0，评测进程默认也用 GPU 0 → OOM**。
   必须在评测侧显式 `CUDA_VISIBLE_DEVICES=1`。

## 下一步

- [ ] Traditional 跑完 → 出 FreeChunker vs Traditional 对比表
- [ ] 重跑 FreeChunker（确认本地 jina 路径下的数字一致）
- [ ] 若要扩到 6 域：加 `--domains all`（不需要重新分块，256 目录已经有全部 6 个域）
- [ ] 若要加 PPL / Margin / Lumber：先跑 `6_chunk_ppl.py` / `8_chunk_margin.py` /
      `5_chunk_lumber.py`（都依赖 vLLM，Lumber 只切了 single-doc + multi-doc 两个域）
