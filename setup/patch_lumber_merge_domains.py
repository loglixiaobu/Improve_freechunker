"""把 merge_lumber_shards.py 的 DOMAINS 从 2 域扩展到 6 域。幂等。"""
import sys

P = "/home1/lh/FreeChunker/setup/merge_lumber_shards.py"
OLD = 'DOMAINS = ["single-doc", "multi-doc"]'
NEW = (
    'DOMAINS = ["single-doc", "multi-doc", "code-repo", "long-icl",\n'
    '           "long-dialogue", "long-structured"]'
)

src = open(P, encoding="utf-8").read()

if "long-structured" in src and "DOMAINS = [" in src and OLD not in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

if OLD not in src:
    print("ANCHOR_NOT_FOUND")
    print("当前 DOMAINS 行:", [l for l in src.splitlines() if l.startswith("DOMAINS")])
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
open(P, "w", encoding="utf-8").write(src)
print("PATCH_OK")
