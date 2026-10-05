#!/usr/bin/env bash
# ============================================================================
# FreeChunker 复现 —— 编码 + 训练 一键编排
#
#   Stage 0  LongBench-v2 按 domain 拆成 6 个子目录（评估脚本按 config 加载）
#   Stage 1  train 语料 8 卡分片编码（part 级断点续跑）
#   Stage 2  val[0:200) + test[200:500) 编码（与原始脚本口径一致）
#   Stage 3  合并 part -> 标准 Dataset 目录（train / val / test）
#   Stage 4  G1 数据门禁（覆盖范围 / 行数 / 形状 / 维度 / 有限值）
#   Stage 5  训练跨粒度编码器（单卡 batch=1，2 epoch = 200000 步，可 --resume）
#
# 设计原则：**每一步都幂等**。已存在的 part 会跳过，合并目录会覆盖，
# 训练带 --resume。所以中途断了（掉线 / 机器重启 / 手动 kill）直接重跑本脚本即可。
#
# 用法：
#   bash setup/run_all.sh                # 全流程
#   bash setup/run_all.sh --from 5       # 从 Stage 5（训练）开始
#   bash setup/run_all.sh --only 1       # 只跑 Stage 1
#   NSHARDS=8 SMOKE=1 bash setup/run_all.sh   # 冒烟（每片 120 篇、训练 30 步不落盘）
#
# 环境变量：
#   FC_ROOT   项目根（默认 ~/FreeChunker）
#   FC_PY     python 解释器（默认 ~/miniconda/envs/freechunker/bin/python）
#   NSHARDS   编码分片数 = 用几张卡（默认 8）
#   SLICE     每个 part 多少篇（默认 1000）
#   SMOKE=1   冒烟：产物写到 _smoke/ 下，不污染正式目录
# ============================================================================
set -uo pipefail

ROOT="${FC_ROOT:-$HOME/FreeChunker}"
PY="${FC_PY:-$HOME/miniconda/envs/freechunker/bin/python}"
NSHARDS="${NSHARDS:-8}"
SLICE="${SLICE:-1000}"
SMOKE="${SMOKE:-0}"

# val/test 的切片范围 —— 与原始脚本一致：
#   train: 全量        val: [0,200)        test: [200,500)
VAL_LO=0;   VAL_HI=200
TEST_LO=200; TEST_HI=500

LOGD="$ROOT/logs"
mkdir -p "$LOGD"

# 冒烟模式：产物全部写到 _smoke/ 下，**绝不污染正式目录**
if [ "$SMOKE" = "1" ]; then
  export FC_VECTOR="${FC_VECTOR:-$ROOT/_smoke/vector}"
  export FC_SAVED="${FC_SAVED:-$ROOT/_smoke/saved_models}"
  LOGD="$ROOT/logs/_smoke"
  mkdir -p "$LOGD"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export FC_ROOT="$ROOT"
# 8 个编码进程并发，每个限制线程数，避免 8×N 线程互相抢
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# 长尾文本会频繁申请/释放大块显存，开这个能显著减少碎片导致的假 OOM
# （torch 2.9 起 PYTORCH_CUDA_ALLOC_CONF 已改名，两个都设上以防版本差异）
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOC_CONF"

cd "$ROOT" || { echo "!! 找不到 $ROOT"; exit 1; }

ts()  { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
hr()  { echo "------------------------------------------------------------------------"; }
die() { say "!! $*"; echo "RUN_ALL_FAILED"; exit 1; }

FROM=1; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done
should_run() {   # should_run <stage_no>
  [ -n "$ONLY" ] && [ "$ONLY" != "$1" ] && return 1
  [ "$1" -lt "$FROM" ] && return 1
  return 0
}

hr
say "FreeChunker 复现编排启动"
say "ROOT=$ROOT"
say "PY=$PY"
say "分片数=$NSHARDS  part大小=$SLICE  SMOKE=$SMOKE"
say "val=[$VAL_LO,$VAL_HI)  test=[$TEST_LO,$TEST_HI)"
say "HF_ENDPOINT=$HF_ENDPOINT"
say "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
hr

[ -x "$PY" ] || die "python 解释器不存在或不可执行: $PY"
"$PY" -c "import torch;assert torch.cuda.is_available();print('torch',torch.__version__,'cuda',torch.cuda.device_count())" \
  || die "torch/CUDA 不可用"
NGPU="$("$PY" -c 'import torch;print(torch.cuda.device_count())')"
say "可见 GPU 数 = $NGPU"
[ "$NSHARDS" -le "$NGPU" ] || die "NSHARDS=$NSHARDS > 可见 GPU=$NGPU"

# ---------------------------------------------------------------- Stage 0
if should_run 0; then
  hr; say "STAGE 0/5  LongBench-v2 按 domain 拆分"
  if "$PY" "train and preprocess/0_prepare_longbench.py" > "$LOGD/0_prepare_longbench.log" 2>&1; then
    tail -n 8 "$LOGD/0_prepare_longbench.log" | sed 's/^/    /'
    say "STAGE 0 OK"
  else
    tail -n 30 "$LOGD/0_prepare_longbench.log" | sed 's/^/    /'
    die "STAGE 0 失败"
  fi
fi

# ---------------------------------------------------------------- Stage 1/2
# 分片编码：每个分片绑一张卡，part 按全局下标命名，天然不撞名、可续跑
encode_sharded() {   # encode_sharded <split> <nshards>
  local split="$1" nshards="$2"
  local -a pids=()
  local k extra=""
  [ "$SMOKE" = "1" ] && extra="--limit 120"

  say "并行编码 split=$split：$nshards 片 -> GPU 0..$((nshards-1))"
  for ((k=0; k<nshards; k++)); do
    local log="$LOGD/encode_${split}_shard${k}.log"
    CUDA_VISIBLE_DEVICES="$k" "$PY" "train and preprocess/1_build_pretrain_datasets.py" \
      --split "$split" --shard "$k" --nshards "$nshards" \
      --slice-size "$SLICE" $extra \
      > "$log" 2>&1 &
    pids+=($!)
  done

  local fail=0
  for ((k=0; k<nshards; k++)); do
    if wait "${pids[$k]}"; then
      say "  shard $k 完成 $(grep -c 'PART_SHARD_DONE' "$LOGD/encode_${split}_shard${k}.log" 2>/dev/null | tr -d '\n') 次 PART_SHARD_DONE"
    else
      say "  !! shard $k 失败，日志尾部："
      tail -n 25 "$LOGD/encode_${split}_shard${k}.log" | sed 's/^/      /'
      fail=1
    fi
  done
  [ "$fail" = 0 ] || die "$split 有分片失败，修好后重跑本脚本（已完成 part 会跳过）"
}

encode_range() {   # encode_range <split> <corpus_split> <lo> <hi> <gpu>
  local split="$1" corpus="$2" lo="$3" hi="$4" gpu="$5" extra=""
  [ "$SMOKE" = "1" ] && extra="--limit 120"
  local log="$LOGD/encode_${split}.log"
  say "编码 $split <- 语料 $corpus [$lo,$hi)  (GPU $gpu)"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PY" "train and preprocess/1_build_pretrain_datasets.py" \
       --split "$split" --corpus-split "$corpus" --start "$lo" --end "$hi" \
       --slice-size "$SLICE" $extra > "$log" 2>&1; then
    grep -E '本片|完成|PART_SHARD_DONE' "$log" | tail -3 | sed 's/^/    /'
  else
    tail -n 25 "$log" | sed 's/^/    /'
    die "$split 编码失败"
  fi
}

if should_run 1; then
  hr; say "STAGE 1/5  train 语料分片编码"
  T0=$(date +%s)
  encode_sharded train "$NSHARDS"
  say "STAGE 1 OK，耗时 $(( ($(date +%s)-T0)/60 )) min"
fi

if should_run 2; then
  hr; say "STAGE 2/5  val / test 编码"
  T0=$(date +%s)
  encode_range val  val "$VAL_LO"  "$VAL_HI"  0
  encode_range test val "$TEST_LO" "$TEST_HI" 1
  say "STAGE 2 OK，耗时 $(( ($(date +%s)-T0)/60 )) min"
fi

# ---------------------------------------------------------------- Stage 3
if should_run 3; then
  hr; say "STAGE 3/5  合并 part -> 标准 Dataset 目录"
  for split in train val test; do
    if "$PY" "train and preprocess/1b_merge_parts.py" --split "$split" --remove-parts \
         > "$LOGD/merge_${split}.log" 2>&1; then
      grep -E '行数|抽检|已保存|已删除|manifest|MERGE_DONE' "$LOGD/merge_${split}.log" | sed 's/^/    /'
    else
      tail -n 30 "$LOGD/merge_${split}.log" | sed 's/^/    /'
      die "STAGE 3 合并 $split 失败"
    fi
  done
  say "STAGE 3 OK"
fi

# ---------------------------------------------------------------- Stage 4
if should_run 4; then
  hr; say "STAGE 4/5  G1 数据门禁"
  G1_EXTRA=""
  [ "$SMOKE" = "1" ] && G1_EXTRA="--allow-partial"
  if "$PY" "setup/g1_check.py" --split train --split val --split test $G1_EXTRA \
       > "$LOGD/g1_check.log" 2>&1; then
    cat "$LOGD/g1_check.log" | sed 's/^/    /'
    say "STAGE 4 OK (G1_PASS)"
  else
    cat "$LOGD/g1_check.log" | sed 's/^/    /'
    die "STAGE 4 失败：编码产物与语料没对齐，**不要**往下训练"
  fi
fi

# ---------------------------------------------------------------- Stage 5
if should_run 5; then
  hr; say "STAGE 5/5  训练跨粒度编码器（单卡，batch=1，2 epoch）"
  TRAIN_EXTRA=""
  if [ "$SMOKE" = "1" ]; then
    TRAIN_EXTRA="--max-steps 30 --no-save"
    say "!! SMOKE 模式：只跑 30 步且不落盘"
  fi
  say "训练日志 -> $LOGD/train.log ；同时写 $ROOT/saved_models/jina-embeddings-v2-small-en/train_log.txt"
  CUDA_VISIBLE_DEVICES=0 "$PY" "train and preprocess/3_train_jina.py" \
    --epochs 2 --lr 1e-4 --seed 0 --eval-interval 1000 --ckpt-interval 10000 \
    --val-samples 200 --resume $TRAIN_EXTRA \
    > "$LOGD/train.log" 2>&1
  rc=$?
  tail -n 30 "$LOGD/train.log" | sed 's/^/    /'
  [ "$rc" = 0 ] || die "STAGE 5 训练退出码 $rc"
  grep -q 'TRAIN_DONE' "$LOGD/train.log" || die "STAGE 5 没看到 TRAIN_DONE，可能被中断（重跑本脚本会 --resume 续上）"
  say "STAGE 5 OK"
fi

hr
say "全流程完成"
hr
echo "RUN_ALL_DONE"
