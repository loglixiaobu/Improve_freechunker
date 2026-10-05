#!/usr/bin/env python3
"""给 PPL(6) / Margin(8) 两个 chunker 打「按域分片 + 逐样本 checkpoint」补丁。

动机
----
PPL / Margin 是全流程最慢的两个（本机实测 Margin ~9.5k char/s，PPL 更慢），
串行跑 4.38 亿字符要 ~13 小时。而且原脚本**只在全部域跑完后才一次性
save_to_disk**，中途崩了整批全丢。

补丁做两件事
------------
1. **按域分片**：读环境变量
       FC_CM_DOMAINS   逗号分隔，只跑这些域（不设=全部）
       FC_CM_SHARD     本 worker 序号（默认 0）
       FC_CM_NSHARDS   总 worker 数（默认 1）
   每个 worker 领 `domains[shard::nshards]`，各自加载自己的模型上自己的卡。

2. **逐样本 checkpoint**：每跑完一个样本就 append 一行 JSONL 到
       LongBench-v2_chunked/<PPL|Margin>/<model_name>/checkpoints/<domain><.shardN>.jsonl
   带 `.shardN` 后缀（nshards>1 时），避免并发 worker 互相覆盖。
   重跑时按 `_id` 去重跳过，断了能续。

3. 输出目录同样带 `.shardN` 后缀，由 `merge_cm_shards.py` 合并。

PPL 是「整批一次性保存」的写法，补丁会把它改造成**逐域保存**（和 Margin 一致），
这样域之间互不连累。
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = "# --- fc-patch: cm sharding ---"

# ---------------------------------------------------------------- 公共片段

HELPER = '''
{marker}
import json as _json
_CM_DOMAINS = [d.strip() for d in _os.environ.get('FC_CM_DOMAINS', '').split(',') if d.strip()]
_CM_SHARD = int(_os.environ.get('FC_CM_SHARD', '0') or '0')
_CM_NSHARDS = int(_os.environ.get('FC_CM_NSHARDS', '1') or '1')
_CM_SUFFIX = ('.shard%d' % _CM_SHARD) if _CM_NSHARDS > 1 else ''
_CM_OUT_ROOT = _os.path.join('{out_root}', model_name)
_CM_CKPT_DIR = _os.path.join(_CM_OUT_ROOT, 'checkpoints')
_os.makedirs(_CM_CKPT_DIR, exist_ok=True)

def _cm_ckpt_path(dn):
    return _os.path.join(_CM_CKPT_DIR, dn + _CM_SUFFIX + '.jsonl')

def _cm_load_done(dn):
    """读回已完成样本的 _id 集合，断点续跑用。"""
    p = _cm_ckpt_path(dn)
    done = {{}}
    if _os.path.isfile(p):
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                done[rec.get('_id')] = rec
    return done

def _cm_append(dn, rec):
    with open(_cm_ckpt_path(dn), 'a', encoding='utf-8') as f:
        f.write(_json.dumps(rec, ensure_ascii=False) + '\\n')
        f.flush()

def _cm_pick(domains_order):
    """按 shard 选择本 worker 要跑的域。"""
    picked = domains_order
    if _CM_DOMAINS:
        picked = [d for d in picked if d in _CM_DOMAINS]
    if _CM_NSHARDS > 1:
        picked = picked[_CM_SHARD::_CM_NSHARDS]
    return picked
'''

# ---------------------------------------------------------------- Margin (8)

MARGIN_OLD_TAIL = '''save_base = './LongBench-v2_chunked/Margin'
new_dataset_root = os.path.join(save_base, model_name)
os.makedirs(new_dataset_root, exist_ok=True)
print("\\n💾 Saving new chunked datasets...")
for dn, ds in new_datasets.items():
    out_dir = os.path.join(new_dataset_root, dn)
    ds.save_to_disk(out_dir)
index_path = os.path.join(new_dataset_root, "dataset_dict.json")
with open(index_path, "w", encoding="utf-8") as f:
    json.dump({"splits": list(new_datasets.keys())}, f, ensure_ascii=False)
print(f"New chunked datasets saved to: {new_dataset_root}")
print("\\n✅ Processing completed!")'''

MARGIN_NEW_TAIL = '''save_base = './LongBench-v2_chunked/Margin'
new_dataset_root = os.path.join(save_base, model_name)
os.makedirs(new_dataset_root, exist_ok=True)
print("\\n💾 Saving new chunked datasets...")
for dn, ds in new_datasets.items():
    out_dir = os.path.join(new_dataset_root, dn + _CM_SUFFIX)
    ds.save_to_disk(out_dir)
index_path = os.path.join(new_dataset_root, "dataset_dict" + _CM_SUFFIX + ".json")
with open(index_path, "w", encoding="utf-8") as f:
    json.dump({"splits": list(new_datasets.keys())}, f, ensure_ascii=False)
print(f"New chunked datasets saved to: {new_dataset_root}")
print("\\n✅ Processing completed!")'''

# ---------------------------------------------------------------- PPL (6)

PPL_OLD_BLOCK = '''new_datasets[task_name] = Dataset.from_list(task_chunk_results)
    print(f"✅ {task_name} chunking completed, generated {len(task_chunk_results)} samples")
new_dataset_dict = DatasetDict(new_datasets)
save_base = './LongBench-v2_chunked/PPL'
os.makedirs(save_base, exist_ok=True)
save_path = os.path.join(save_base, model_name)
print("\\n💾 Saving new chunked datasets...")
new_dataset_dict.save_to_disk(save_path)
print(f"New chunked datasets saved to: {save_path}")
print("\\n✅ Processing completed!")'''

PPL_NEW_BLOCK = '''new_datasets[task_name] = Dataset.from_list(task_chunk_results)
    print(f"✅ {task_name} chunking completed, generated {len(task_chunk_results)} samples")
    out_base = os.path.join('./LongBench-v2_chunked/PPL', model_name)
    os.makedirs(out_base, exist_ok=True)
    new_datasets[task_name].save_to_disk(os.path.join(out_base, task_name + _CM_SUFFIX))
    with open(os.path.join(out_base, "dataset_dict" + _CM_SUFFIX + ".json"), 'w', encoding='utf-8') as f:
        json.dump({"splits": list(new_datasets.keys())}, f, ensure_ascii=False)
print("\\n✅ Processing completed!")'''


def patch_margin(text: str) -> str:
    helper = HELPER.format(marker=MARKER, out_root='./LongBench-v2_chunked/Margin')
    if MARKER in text:
        print("  [margin] 已打过补丁，跳过")
        return text

    # 1) 注入 helper（放在 selected_domains 之前）
    anchor = "selected_domains = ['single-doc','multi-doc','code-repo','long-dialogue','long-icl','long-structured']"
    if anchor not in text:
        raise SystemExit("[margin] 找不到 selected_domains 锚点")
    text = text.replace(anchor, helper.strip() + "\n\n" + anchor, 1)

    # 2) selected_domains 改为受 _cm_pick 控制
    text = text.replace(
        anchor,
        "selected_domains = _cm_pick(['single-doc','multi-doc','code-repo',"
        "'long-dialogue','long-icl','long-structured'])\n"
        "print(f'[cm] shard={_CM_SHARD}/{_CM_NSHARDS} domains={selected_domains}')",
        1,
    )

    # 3) 样本循环里加 checkpoint（在 task_chunk_results.append(new_sample) 之后）
    old_append = "            task_chunk_results.append(new_sample)"
    if old_append not in text:
        raise SystemExit("[margin] 找不到 append 锚点")
    text = text.replace(
        old_append,
        old_append + "\n            _cm_append(dn, new_sample)",
        1,
    )

    # 4) 输出目录加 shard 后缀
    if MARGIN_OLD_TAIL not in text:
        raise SystemExit("[margin] 找不到尾部保存块")
    text = text.replace(MARGIN_OLD_TAIL, MARGIN_NEW_TAIL, 1)
    return text


def patch_ppl(text: str) -> str:
    helper = HELPER.format(marker=MARKER, out_root='./LongBench-v2_chunked/PPL')
    if MARKER in text:
        print("  [ppl] 已打过补丁，跳过")
        return text

    # 1) 注入 helper（放在 "root = './LongBench-v2'" 之前）
    anchor = "root = './LongBench-v2'"
    if anchor not in text:
        raise SystemExit("[ppl] 找不到 root 锚点")
    text = text.replace(anchor, helper.strip() + "\n\n" + anchor, 1)

    # 2) 加载后按 shard 过滤（datasets 是 DatasetDict）
    old_load = ("datasets = DatasetDict({d: load_from_disk(_os.path.join(_SPLIT_DIR, d))\n"
                "                        for d in sorted(_os.listdir(_SPLIT_DIR)) "
                "if _os.path.isdir(_os.path.join(_SPLIT_DIR, d))})")
    if old_load not in text:
        raise SystemExit("[ppl] 找不到 datasets 加载块")
    text = text.replace(
        old_load,
        "_all_domains = sorted([d for d in _os.listdir(_SPLIT_DIR) "
        "if _os.path.isdir(_os.path.join(_SPLIT_DIR, d))])\n"
        "_picked = _cm_pick(_all_domains)\n"
        "print(f'[cm] shard={_CM_SHARD}/{_CM_NSHARDS} domains={_picked}')\n"
        "datasets = DatasetDict({d: load_from_disk(_os.path.join(_SPLIT_DIR, d)) for d in _picked})",
        1,
    )

    # 3) 循环开头按 _id 去重（断点续跑）
    old_loop = "    for i, sample in enumerate(task_dataset):\n        context = sample['context']"
    if old_loop not in text:
        raise SystemExit("[ppl] 找不到样本循环锚点")
    text = text.replace(
        old_loop,
        "    _done = _cm_load_done(task_name)\n"
        "    for i, sample in enumerate(task_dataset):\n"
        "        if sample['_id'] in _done:\n"
        "            task_chunk_results.append(_done[sample['_id']])\n"
        "            continue\n"
        "        context = sample['context']",
        1,
    )

    # 4) 循环里加 checkpoint
    old_append = "        task_chunk_results.append(new_sample)"
    if old_append not in text:
        raise SystemExit("[ppl] 找不到 append 锚点")
    text = text.replace(
        old_append,
        old_append + "\n        _cm_append(task_name, new_sample)",
        1,
    )

    # 5) 尾部：整批一次保存 -> 逐域保存 + shard 后缀
    if PPL_OLD_BLOCK not in text:
        raise SystemExit("[ppl] 找不到尾部保存块")
    text = text.replace(PPL_OLD_BLOCK, PPL_NEW_BLOCK, 1)
    return text


def main():
    targets = {
        "6_chunk_ppl.py": patch_ppl,
        "8_chunk_margin.py": patch_margin,
    }
    for fn, fn_patch in targets.items():
        path = os.path.join(ROOT, "train and preprocess", fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        new = fn_patch(text)
        if new != text:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            print(f"  [ok] patched {fn}")
        else:
            print(f"  [--] {fn} unchanged")
    print("CM_PATCH_DONE")


if __name__ == "__main__":
    main()
