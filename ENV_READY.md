# 环境 / 数据集 / 模型 已就绪（2026-10-03）

复现目标：`jina-embeddings-v2-small-en`，训练粒度按代码用 **`[2, 4]`**。

自检结果：**13 项全通过，0 失败**（完整输出见 `logs/verify_report.txt`）。

---

## 1. conda 环境（新建，未复用任何已有环境）

```
环境名：freechunker
路径：  /home1/lh/miniconda/envs/freechunker      （11 GB）
Python：3.10.22
```

激活：

```bash
source /home1/lh/miniconda/etc/profile.d/conda.sh
conda activate freechunker
```

或者直接用绝对路径（不需要 activate）：

```bash
/home1/lh/miniconda/envs/freechunker/bin/python <脚本>
```

版本与已验证可用的 `dcs` 环境逐一对齐：

| 包 | 版本 | 包 | 版本 |
|---|---|---|---|
| torch | 2.9.1+cu128 | datasets | 4.4.1 |
| torchvision | 0.24.1 | accelerate | 1.11.0 |
| transformers | 4.57.1 | numpy | 2.2.6 |
| tokenizers | 0.22.1 | pandas | 2.3.3 |
| sentence-transformers | 3.2.0 | scikit-learn | 1.7.2 |
| vllm | 0.16.0 | matplotlib | 3.10.7 |
| openai | 2.24.0 | pyarrow | 22.0.0 |
| huggingface-hub | 0.36.0 | sentencepiece | 0.2.1 |
| einops | 0.8.1 | nltk | 3.9.1 |
| json_repair | 0.63.5 | safetensors | 0.6.2 |

GPU 自检：8 × RTX 3090，`sm_86`，`cuda:0` 实算 matmul 正常。

> **注意**：`json_repair` 是 `test/test_freechunker.py` 的依赖，原 `dcs` 环境里没有，新环境已装上。

---

## 2. 目录结构（模型和数据集都在项目内）

```
~/FreeChunker/
├── models/                                   共 20 GB
│   ├── jina-embeddings-v2-small-en/          500 MB   teacher / 句级编码器（dim 512, max_len 8192）
│   ├── FreeChunk-jina/                       1.2 GB   官方 checkpoint（311.8M 参数，含 training_losses.json）
│   ├── bge-m3/                               2.2 GB   跨粒度编码器的初始化权重（注意是 pytorch_model.bin）
│   └── Qwen3-8B/                              16 GB   生成模型（5 shard，36 层，hidden 4096，max_pos 40960）
├── datasets/                                 3.9 GB
│   ├── FreeChunk-corpus/                     3.4 GB   train 7 shard + val 1 shard
│   │   ├── train/   data-00000..00006-of-00007.arrow   （100,000 篇）
│   │   └── val/     data-00000-of-00001.arrow          （9,880 篇）
│   └── LongBench-v2/                         444 MB   data.json（503 题，6 个 domain）
├── setup/                                    安装与自检脚本（留档）
│   ├── setup_env.sh                          建环境（含并行启动下载）
│   ├── download_assets.py                    下载全部模型/数据集
│   └── verify_env.py                         就绪自检
├── logs/
│   ├── setup_env.log                         环境安装日志
│   ├── download_assets.log                   下载日志
│   └── verify_report.txt                     ★ 就绪自检完整报告
├── vector/  saved_models/                    空，留给 Phase 1 / 2 的产物
├── 复现流程表.md / .html                      复现计划（含论文 vs 代码对照、验收门、坑清单）
└── src/ test/ baseline/ train and preprocess/  原仓库代码（未改动）
```

**重要**：下载时全部指定了 `local_dir`，所以**没有走 HF 缓存**，模型和数据集都实实在在躺在项目目录里。
`/home1` 剩余 **273 G**（下载 + 环境共占用约 38 G）。

---

## 3. 数据自检的关键数字

| 项 | 实测值 | 说明 |
|---|---|---|
| train 篇数 | **100,000** | 与官方 200,000 步 ÷ 2 epoch ÷ batch 1 完全吻合 |
| val 篇数 | **9,880** | 够用（脚本取 0:200 作 val、200:500 作 test） |
| 每篇 token | 恒为 **8,000** | `original_token_count` 字段 |
| 每篇句数 | mean **124.0** / p50 126 / max 230 | 是**真句子**（≈64 token/句），不是 256-token 块 |
| `[2,4]` span 数 | 代码复算 == 独立复算 | 117 句 → 174 条 span |
| 每篇向量数 | mean **308.1** 条 | 124.0 句 + 184.2 span |
| 全量编码量 | ≈ **30.8 M** 条 | ×100,000 |

LongBench-v2 的 6 个 domain（题数）：Single-Document QA 175、Multi-Document QA 125、
Long In-context Learning 81、Code Repository 50、Long-dialogue 39、Long Structured Data 33。

官方训练曲线（G2 验收靶子，取自 `models/FreeChunk-jina/training_losses.json`）：
`total_steps=200000`、前 1000 步均值 **0.4497**、末 1000 步均值 **0.0109**、`final_val_loss` **0.0103**。

---

## 4. 还没动的 5 处代码问题（等你确认后再改）

原仓库代码**原样未改**。这 5 处是跑之前必须修的，其中 3 处会直接报错：

| # | 问题 | 文件 | 是否阻断 |
|---|---|---|---|
| 1 | 默认 tokenizer 硬编码 `/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B`，服务器上不存在；**而它决定句子切分粒度** | `baseline/traditional_chunking.py` | 🔴 阻断 |
| 2 | 本地语料**没有根 `dataset_dict.json`**，`load_dataset` 和 `load_from_disk(根)` 都会失败，必须按 split 子目录加载 | `train and preprocess/1_build_pretrain_datasets.py` | 🔴 阻断 |
| 3 | 造数据脚本把 part_* 存到 `./vector/jina-.../part_N`，训练脚本却读 `./vector/jina-.../train` | `1_build_pretrain_datasets.py` + `3_train_jina.py` | 🔴 阻断 |
| 4 | `src/` 没有 `__init__.py`，直接 `python "train and preprocess/3_train_jina.py"` 时 `sys.path[0]` 是脚本目录 → `from src...` 失败 | 运行方式 | 🟠 需 `PYTHONPATH=.` |
| 5 | `dataset_path` 是相对路径 `../Data/LongBench-v2`；`encoder_model_path` 与训练脚本实际输出路径也不一致 | `test/test_freechunker.py` | 🟠 |

另外，训练粒度按你的要求**以代码为准用 `[2, 4]`** —— 代码里 `generate_shifted_matrix` 的默认值本来就是
`[2, 4]`，官方权重 `config.json` 里也是 `max_power: 4`，所以这一点不需要改代码，只要不主动传别的粒度即可。

---

## 5. 下一步

确认无误后，Phase 1 的第一步是：
1. 把 LongBench-v2 的 `data.json` 按 `domain` 拆成 6 个子目录（因为脚本按 domain 当 config 加载）；
2. 修上面 5 处；
3. 8 卡分片跑编码（预计 1.3–1.5 h，产物约 63 GB）。
