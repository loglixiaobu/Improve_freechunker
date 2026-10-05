#!/usr/bin/env bash
# 一眼看全：vLLM / GPU / 各 chunker 进度 / 评测进度
set -uo pipefail
echo "==================== $(date '+%H:%M:%S') ===================="
echo "--- vLLM ---"
for p in 8888 8889; do printf "  %s: " $p; curl -s -m 3 -o /dev/null -w "%{http_code}\n" http://localhost:$p/health; done
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- tmux ---"
tmux ls 2>/dev/null | sed 's/^/  /'
echo "--- semantic shards ---"
for g in 2 3 4 5 6 7; do
  printf "  g%s: " $g
  grep -o "[0-9]*/[0-9]* \[[0-9:]*<[0-9:]*, [0-9.]*s/sample\]" ~/FreeChunker/logs/chunk_semantic_g$g.log 2>/dev/null | tail -1
  echo
done
echo "--- lumber workers (checkpoint 行数) ---"
wc -l ~/FreeChunker/LongBench-v2_chunked/Lumber/*/checkpoints/*.jsonl 2>/dev/null | sed 's/^/  /'
echo "--- semantic checkpoints ---"
wc -l ~/FreeChunker/LongBench-v2_chunked/Semantic/*/checkpoints/*.jsonl 2>/dev/null | sed 's/^/  /'
echo "--- ppl / margin ---"
for n in ppl margin; do printf "  %s: " $n; grep -o "[0-9]*/[0-9]* \[[0-9:]*<[0-9:]*, [0-9.]*s/sample\]" ~/FreeChunker/logs/chunk_$n.log 2>/dev/null | tail -1; echo; done
echo "--- eval ---"
ls -t ~/FreeChunker/eval_results/raw_*.jsonl 2>/dev/null | head -3 | sed 's/^/  /'
echo "=================================================="
