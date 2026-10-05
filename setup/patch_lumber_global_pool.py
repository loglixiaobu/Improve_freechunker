#!/usr/bin/env python3
"""给 5_chunk_lumber.py 加「全局已完成池」，让 worker 数可以随意变。

问题
----
现有 checkpoint 是 **每个 shard 一个文件**：`<domain>.shardN.jsonl`，
worker 只从**自己那个文件**里读已完成 `_id`。一旦把 workers 从 6 改成 48，
`i % nshards` 的映射全变，老 shard 文件里的 25 条对不上新 worker 的份额
→ 这 25 条会被重跑（结果不错，但白烧算力）。

做法
----
1. 新增全局池 `checkpoints/_done_global.jsonl`（append-only）。
   每个样本跑完，**同时**写自己的 shard 文件**和**全局池。
2. 启动时，先读全局池，把这些 `_id` 全部预填进 `processed_samples`，
   实现「跨 shard 断点续跑」。
3. 提供 `tools/build_lumber_done_pool.py` 把历史 shard 文件灌进全局池。

这样 worker 数从 6 → 48 → 任意，都不会重跑已完成的样本。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "train and preprocess", "5_chunk_lumber.py")
MARKER = "# fc-patch: lumber global done pool"

SNIPPET = '''
{marker}
_GLOBAL_DONE_PATH = _os.path.join(checkpoints_root, "_done_global.jsonl")

def _lmb_load_global_done():
    """读全局已完成池 -> {{_id: record}}，跨 shard 断点续跑用。"""
    done = {{}}
    if _os.path.isfile(_GLOBAL_DONE_PATH):
        with open(_GLOBAL_DONE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except Exception:
                    continue
                if obj.get("_id") is not None:
                    done[obj["_id"]] = obj
    return done

def _lmb_append_global(rec):
    with open(_GLOBAL_DONE_PATH, "a", encoding="utf-8") as f:
        f.write(_json.dumps(rec, ensure_ascii=False) + "\\n")
        f.flush()
'''

# 现有：本地 shard checkpoint 读取之后，再并入全局池
OLD_LOCAL_LOAD = '''    processed_samples = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    _id = obj.get("_id")
                    if _id is not None:
                        processed_samples[_id] = obj
                except:
                    pass
'''

NEW_LOCAL_LOAD = '''    processed_samples = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    _id = obj.get("_id")
                    if _id is not None:
                        processed_samples[_id] = obj
                except:
                    pass
    # fc-patch: lumber global done pool -- 并入全局池
    _global_done = _lmb_load_global_done()
    for _gid, _grec in _global_done.items():
        processed_samples.setdefault(_gid, _grec)
    print(f"[lumber] 本地 checkpoint {{len(processed_samples) - len(_global_done)}} 条 + 全局池 {{len(_global_done)}} 条 = 待跳过 {{len(processed_samples)}} 条")
'''

# 现有：写本地 shard checkpoint 之后，同时写全局池
OLD_APPEND = '''            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(new_sample, ensure_ascii=False) + "\\n")
            overall_pbar.update(1)'''

NEW_APPEND = '''            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(new_sample, ensure_ascii=False) + "\\n")
            _lmb_append_global(new_sample)  # fc-patch: lumber global done pool
            overall_pbar.update(1)'''


def main():
    with open(TARGET, "r", encoding="utf-8") as f:
        text = f.read()

    if MARKER in text:
        print("[lumber] 已打过全局池补丁，跳过")
        print("LUMBER_POOL_PATCH_DONE")
        return

    # 1) 注入 helper（放在 checkpoints_root 创建之后）
    anchor = 'checkpoints_root = os.path.join(new_dataset_root, "checkpoints")\nos.makedirs(checkpoints_root, exist_ok=True)'
    if anchor not in text:
        raise SystemExit("[lumber] 找不到 checkpoints_root 锚点")
    text = text.replace(anchor, anchor + "\n" + SNIPPET.strip().format(marker=MARKER), 1)

    # import json as _json 需要存在
    if "import json as _json" not in text:
        text = text.replace("import json\n", "import json\nimport json as _json\n", 1)

    # 2) 本地 checkpoint 读取 -> 并入全局池
    if OLD_LOCAL_LOAD not in text:
        raise SystemExit("[lumber] 找不到本地 checkpoint 读取块")
    text = text.replace(OLD_LOCAL_LOAD, NEW_LOCAL_LOAD, 1)

    # 3) 写本地 -> 同时写全局
    if OLD_APPEND not in text:
        raise SystemExit("[lumber] 找不到 checkpoint append 块")
    text = text.replace(OLD_APPEND, NEW_APPEND, 1)

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(text)
    print("[ok] patched 5_chunk_lumber.py")
    print("LUMBER_POOL_PATCH_DONE")


if __name__ == "__main__":
    main()
