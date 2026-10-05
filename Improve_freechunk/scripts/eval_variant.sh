#!/usr/bin/env bash
# 评测一个变体（默认只跑 FreeChunker，6 域全量），产物落到 results/<exp-id>/。
#
# 用法:
#   bash scripts/eval_variant.sh <exp-id> [KEY=VAL ...]
#
# 例:
#   bash scripts/eval_variant.sh I1_gran_norm FC_GRAN_NORM=zscore
#   bash scripts/eval_variant.sh baseline
#
# 说明:
#   * 额外 KEY=VAL 会作为环境变量传给评测进程（模块通过读环境变量开关）
#   * 同时跑 Traditional(256) 作为锚点，用于确认改动没泄漏到检索层之外
#   * 结果目录里会留一份 cmd.txt 记录完整命令，便于复现
set -uo pipefail

ROOT="$HOME/FreeChunker"
PY=/home1/lh/miniconda/envs/freechunker/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP="${1:-baseline}"; shift || true
EXTRA_ENV=("$@")

OUT="$HERE/results/$EXP"
mkdir -p "$OUT"

echo "===== exp=$EXP ====="
echo "extra env: ${EXTRA_ENV[*]:-（无）}"

# ---- 记录完整命令，便于复现 ----
{
  echo "# $(date '+%F %T')"
  echo "bash scripts/eval_variant.sh $EXP ${EXTRA_ENV[*]:-}"
  echo "# 额外环境变量:"
  for kv in "${EXTRA_ENV[@]:-}"; do [ -n "$kv" ] && echo "export $kv"; done
} > "$OUT/cmd.txt"

# ---- 环境 ----
export PYTHONPATH="$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export FC_EVAL_MAX_CONTEXT=32768
export FC_MAX_SENTENCES=4000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for kv in "${EXTRA_ENV[@]:-}"; do [ -n "$kv" ] && export "$kv"; done

cd "$ROOT"

# ---- vLLM 健康检查（两个实例）----
for p in 8888 8889; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$p/health" || true)
  if [ "$code" != "200" ]; then
    echo "!! vLLM :$p 未就绪，请先起服务："
    echo "   tmux new-session -d -s vllm  \"... FC_VLLM_GPU=0 FC_VLLM_PORT=8888 bash setup/start_vllm.sh\""
    exit 2
  fi
done
echo "vLLM 双实例就绪"

# ---- 跑评测 ----
# 先清掉上一次的 raw，避免新旧混在一起
rm -f "$ROOT"/eval_results/raw_*.jsonl

echo "--- FreeChunker × 6 域 ---"
$PY setup/run_eval_parallel.py \
    --methods freechunker --domains all --nshards 6 \
    2>&1 | tee "$OUT/eval_freechunker.log" | tail -20
rc_fc=${PIPESTATUS[0]}

echo "--- Traditional(256) 锚点 ---"
$PY setup/run_eval_parallel.py \
    --methods traditional --domains all --nshards 6 \
    2>&1 | tee "$OUT/eval_traditional.log" | tail -10
rc_tr=${PIPESTATUS[0]}

# ---- 收产物 ----
mv "$ROOT"/eval_results/raw_*.jsonl "$OUT"/ 2>/dev/null || true
echo "raw 文件: $(ls -1 "$OUT"/raw_*.jsonl 2>/dev/null | wc -l) 个"

# ---- 生成对照表 ----
$PY "$HERE/scripts/make_gap.py" --eval-dir "$OUT" --out "$OUT/gap.md" || true

echo "===== 完成，产物在 $OUT ====="
[ "$rc_fc" = "0" ] && [ "$rc_tr" = "0" ] || echo "!! 有任务非零退出，请看日志"
