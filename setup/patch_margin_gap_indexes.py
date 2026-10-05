#!/usr/bin/env python3
"""给 `train and preprocess/8_chunk_margin.py` 加 FC_CM_INDEXES / FC_CM_SKIP 支持。

目的
----
`shard_margin_longtail.py` v2 的「缺口续切」需要显式指定样本下标（而不是
`range(shard, n, nshards)` 这种等距划分）。本脚本在原有的 cm-sharding 块里
插入两个环境变量：

  FC_CM_INDEXES  逗号分隔的**显式下标**列表，优先于 shard/nshards
  FC_CM_SKIP     逗号分隔的**跳过下标**（已被别的在跑 worker 认领的）

替换范围：
  1. `_CM_SUFFIX` 计算后，追加 `_CM_INDEXES` / `_CM_SKIP` 的解析
  2. `_cm_keep_idx(n)` 函数体：优先返回显式下标
  3. `_cm_ckpt_path(dn)`：续切时用独立文件名，避免与在跑 worker 争锁
     （用自己的 suffix，如 `.gap<tag>`）

幂等：重复运行不会重复插入（有标记检查）。
"""
import re
import sys

P = "train and preprocess/8_chunk_margin.py"

with open(P, encoding="utf-8") as f:
    src = f.read()

if "FC_CM_INDEXES" in src:
    print("[patch] 已包含 FC_CM_INDEXES，跳过")
    sys.exit(0)

# ---------- 1) 解析环境变量 ----------
anchor = """_CM_SUFFIX = ('.shard%d' % _CM_SHARD) if _CM_NSHARDS > 1 else ''"""
assert anchor in src, "找不到 _CM_SUFFIX 锚点"

ins = anchor + """
# --- fc-patch: gap reshard (显式下标) ---
_CM_INDEXES = [int(x) for x in _os.environ.get('FC_CM_INDEXES', '').split(',') if x.strip()]
_CM_SKIP = set(int(x) for x in _os.environ.get('FC_CM_SKIP', '').split(',') if x.strip())
_CM_GAP_TAG = _os.environ.get('FC_CM_GAP_TAG', '').strip()
if _CM_GAP_TAG:
    _CM_SUFFIX = '.gap' + _CM_GAP_TAG"""
src = src.replace(anchor, ins, 1)

# ---------- 2) _cm_keep_idx 支持显式下标 ----------
old_keep = """def _cm_keep_idx(n):
    \"\"\"本分片在长度 n 的域里应取哪些下标。\"\"\"
    if _CM_NSHARDS <= 1:
        return None
    return set(range(_CM_SHARD, n, _CM_NSHARDS))"""
assert old_keep in src, "找不到 _cm_keep_idx 锚点"

new_keep = """def _cm_keep_idx(n):
    \"\"\"本分片在长度 n 的域里应取哪些下标。

    优先级：FC_CM_INDEXES（显式） > FC_CM_SHARD/FC_CM_NSHARDS（等距）。
    两者都用 FC_CM_SKIP 扣除「已被其他在跑 worker 认领」的下标。
    \"\"\"
    if _CM_INDEXES:
        return set(i for i in _CM_INDEXES if i < n and i not in _CM_SKIP)
    if _CM_NSHARDS <= 1:
        return None
    return set(range(_CM_SHARD, n, _CM_NSHARDS)) - _CM_SKIP"""
src = src.replace(old_keep, new_keep, 1)

with open(P, "w", encoding="utf-8") as f:
    f.write(src)

print("[patch] 已插入 FC_CM_INDEXES / FC_CM_SKIP 支持")
print("  _CM_INDEXES:", "FC_CM_INDEXES" in src)
print("  _CM_SKIP   :", "FC_CM_SKIP" in src)
print("  gap suffix :", "_CM_GAP_TAG" in src)
