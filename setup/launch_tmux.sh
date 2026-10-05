#!/usr/bin/env bash
# 在 tmux 里无人值守地跑 setup/run_all.sh。
#
# 用法：
#   bash setup/launch_tmux.sh
#   FC_SESSION=fc2 bash setup/launch_tmux.sh
#   NSHARDS=8 SMOKE=1 bash setup/launch_tmux.sh     # 冒烟（几十秒，用来验证链路）
#
# 之后：
#   tmux attach -t fc        # 看实时输出（Ctrl-b d 脱离）
#   tail -f logs/run_all.log # 只看日志
set -euo pipefail

ROOT="${FC_ROOT:-$HOME/FreeChunker}"
SESSION="${FC_SESSION:-fc}"
cd "$ROOT"
mkdir -p logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "!! tmux 会话 '$SESSION' 已存在，先看/关掉它："
  tmux ls
  echo "   tmux attach -t $SESSION    # 进去看"
  echo "   tmux kill-session -t $SESSION"
  exit 1
fi

# 把环境变量显式带进 tmux（tmux 起的是新 shell，父 shell 的 export 不一定带过去）
ENVS="FC_ROOT='$ROOT'"
for v in FC_PY NSHARDS SLICE BS SMOKE HF_ENDPOINT OMP_NUM_THREADS; do
  eval "val=\${$v:-}"
  [ -n "$val" ] && ENVS="$ENVS $v='$val'"
done

tmux new-session -d -s "$SESSION" -n run \
  "cd '$ROOT' && $ENVS bash setup/run_all.sh 2>&1 | tee -a logs/run_all.log; \
   echo; echo '=== run_all.sh 已退出（会话保留，方便看现场）==='; exec bash"

echo "已在 tmux 会话 '$SESSION' 中启动。"
tmux ls
echo
echo "  查看实时输出 : tmux attach -t $SESSION   （Ctrl-b 然后 d 脱离）"
echo "  只看日志     : tail -f $ROOT/logs/run_all.log"
echo "  看某分片     : tail -f $ROOT/logs/encode_train_shard0.log"
