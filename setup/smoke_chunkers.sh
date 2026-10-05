#!/usr/bin/env bash
# 每个 chunker 跑 90 秒冒烟，看是否能在前几个样本上正常工作（有异常立刻暴露）
set -uo pipefail
source ~/miniconda/etc/profile.d/conda.sh
conda activate freechunker
cd ~/FreeChunker
export PYTHONPATH=~/FreeChunker
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs

smoke() {
  local name="$1" script="$2" gpu="$3" secs="${4:-90}"
  echo "########## SMOKE $name (gpu=$gpu, ${secs}s) ##########"
  CUDA_VISIBLE_DEVICES="$gpu" timeout "$secs" python "train and preprocess/$script" 2>&1 \
    | tail -40
  echo "########## END $name rc=${PIPESTATUS[0]} ##########"
}

case "${1:-all}" in
  semantic) smoke semantic 7_chunk_semantic.py 5 ;;
  lumber)   smoke lumber   5_chunk_lumber.py   2 120 ;;
  ppl)      smoke ppl      6_chunk_ppl.py      3 ;;
  margin)   smoke margin   8_chunk_margin.py   4 ;;
  all)
    smoke semantic 7_chunk_semantic.py 5 &
    smoke ppl      6_chunk_ppl.py      3 &
    wait
    smoke lumber   5_chunk_lumber.py   2 120
    smoke margin   8_chunk_margin.py   4
    ;;
esac
echo "SMOKE_DONE"
