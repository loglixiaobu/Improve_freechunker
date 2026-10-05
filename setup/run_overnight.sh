#!/usr/bin/env bash
# 夜间流水线启动包装：等 Margin -> 合并 -> 校验 -> 起 vLLM -> 评测 -> 汇总
# 由 tmux 会话 `night` 持有，SSH 断开也不受影响。
set -uo pipefail

cd "$HOME/FreeChunker" || exit 1

source ~/miniconda/etc/profile.d/conda.sh
conda activate freechunker

export PYTHONPATH="$HOME/FreeChunker"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

echo "===== 启动 $(date '+%F %T') ====="
/home1/lh/miniconda/envs/freechunker/bin/python setup/overnight_margin_eval.py
rc=$?
echo "===== 结束 $(date '+%F %T')  rc=$rc ====="
echo "NIGHT_EXIT=$rc"
exit $rc
