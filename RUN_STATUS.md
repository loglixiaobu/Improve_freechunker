# FreeChunker 复现 · 运行状态

> 2026-10-03 21:44 启动 · 服务器 `210.77.30.42` · 项目 `~/FreeChunker`

---

## 现在在跑什么

tmux 会话 **`fc`**，里面是 `bash setup/run_all.sh`（5 阶段，全幂等）。

```bash
tmux attach -t fc                    # 看实时输出（Ctrl-b 然后 d 脱离）
tail -f ~/FreeChunker/logs/run_all.log          # 只看总日志
tail -f ~/FreeChunker/logs/encode_train_shard0.log   # 看某个分片
tmux ls                              # 确认会话还活着
```

## 进度与预算

| 阶段 | 内容 | 实测/预估 |
|---|---|---|
| 0 | LongBench-v2 拆 domain | < 1 min |
| 1 | train 8 卡分片编码（13 个 block/片） | **实测 810–849 向量/s/卡 → ~1.35h** |
| 2 | val[0:200) + test[200:500) 编码 | ~3 min |
| 3 | 合并 part → `vector/jina-embeddings-v2-small-en/{train,val,test}` | ~15 min（写 63GB） |
| 4 | G1 数据门禁 | ~2 min |
| 5 | 训练跨粒度编码器（GPU0，batch=1，2 epoch） | **实测 145 ms/step → ~8h** |

**总计约 9.5 小时**，预计 **10-04 早上 7:15 前后**跑完。

## 早上起来先看这四条

```bash
cd ~/FreeChunker
tail -3 logs/run_all.log                       # 有没有 RUN_ALL_DONE
grep -E "G1_PASS|G1_FAIL" logs/g1_check.log     # 数据门禁
tail -20 logs/train.log                        # 训练进度 / G2 对照
ls -la saved_models/jina-embeddings-v2-small-en/  # 产物
```

**G2 验收靶子**（官方 `models/FreeChunk-jina/training_losses.json`）：

| 指标 | 官方 | 本次 |
|---|---|---|
| total_steps | 200000 | 训练结束会打印 |
| 前 1000 步均值 | 0.4497 | 会打印对照 |
| 末 1000 步均值 | 0.0109 | 会打印对照 |
| final_val_loss | 0.0103 | 会打印对照 |

## 中断了怎么办

**直接重跑，不用清理任何中间产物。**

```bash
bash setup/run_all.sh                # 全流程（已完成的部分会自动跳过）
bash setup/run_all.sh --from 5       # 只续训练（--resume 从 ckpt_latest.pt 续）
```

- 编码：`part_*` 已存在的 block 自动跳过
- 合并：目标目录已存在会覆盖
- 训练：每 10000 步存 `saved_models/jina-embeddings-v2-small-en/ckpt_latest.pt`
  （含 model + optimizer + scheduler + 步数 + loss 历史 + RNG 状态）

## 这次改了什么（相对原始仓库）

| 文件 | 改动 |
|---|---|
| `src/paths.py`（新） | 所有路径的唯一来源，可用 `FC_ROOT/FC_VECTOR/FC_SAVED` 覆盖 |
| `src/__init__.py`（新） | 让 `src` 成为正式包，脚本内 `sys.path.insert` 到仓库根 |
| `baseline/traditional_chunking.py` | 去掉硬编码的 `/share/home/ecnuzwx/...`，改成 显式参数 → 项目内 Qwen3-8B → hub 三级回退 |
| `train and preprocess/0_prepare_longbench.py`（新） | LongBench-v2 只有单个 `data.json`，拆成 6 个 domain 子目录 + `dataset_dict.json` |
| `train and preprocess/1_build_pretrain_datasets.py` | 分片/切片、长度分桶 batch、OOM 二分重试、每分片写 manifest、`--corpus-split` |
| `train and preprocess/1b_merge_parts.py`（新） | 补上原仓库缺的「part → 标准 Dataset 目录」这一步（原仓库两条路径对不上，必崩） |
| `train and preprocess/3_train_jina.py` | 断点续跑、`with_format("numpy")`、`Agg` 后端、显式 per-epoch 置换 |
| `setup/g1_check.py`（新） | G1 数据门禁 |
| `setup/run_all.sh`（新） | 5 阶段编排 |
| `setup/launch_tmux.sh`（新） | tmux 无人值守启动 |

**超参没有动**：AdamW lr=1e-4、batch=1、2 epoch、cosine + 前 1/3 步 warmup、
每 1000 步用 200 条 val 样本评估、粒度 `[2,4]` —— 与原始脚本逐条一致。

## 冒烟测试时抓到的 3 个真 bug

1. **`Features` 少一层** → `TypeError: Couldn't cast array of type list<item: float> to float`。
   一行值是 `[n, 512]` 的二维表，得写 `Sequence(Sequence(Value("float32")))`。
2. **固定 batch_size 撞长尾 → 8 分片集体 CUDA OOM**。
   语料里句子最长 **3471 token**、chunk 最长 **7078 token**，注意力 `B*H*L²*4`，
   B=32 时单次 softmax 要 51 GB。原仓库写死 `batch_size=8` 不是偷懒。
   → 改成按长度分桶（`B*Lmax² ≤ 8e7`）+ OOM 二分重试。
3. **`ds[i]` 慢到占整步 34%**：HF datasets 默认把 Arrow 转 Python 嵌套列表，61.9 ms/步；
   `.with_format("numpy")` → 2.5 ms（25×），20 万步省 **3.3 小时**，数值逐位不变。

## 一处差点白跑 2 小时的口径

原仓库的 val/test 是**切片不是全量**：val = `[0:200)`、test = `[200:500)`（用 validation 语料）。
我原本打算全量编 9,880 篇 val，纯浪费。已在 `run_all.sh` 里写死成常量。

## 磁盘

- `/home1` 启动前剩 **272G**；全量 part 约 63G，合并再要 63G，**峰值约 126G**
- 合并完 `--remove-parts` 自动回收，之后训练 checkpoint 约 4G
- 全程最低水位预计 **~140G**，安全

## 数据事实（实测）

- train **100,000** 篇（7 shards × 14,286），val 语料 **9,880** 篇
- 每篇恒 8000 token，平均 **124.0** 句/篇
- 粒度 `[2,4]` → 平均 **184.2** span/篇 → 每篇 **308.1** 条向量 → 全量 **30.8M** 条
- 向量落盘 ≈ **63 GB**（float32，512 维）
