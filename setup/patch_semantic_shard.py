"""让 7_chunk_semantic.py 支持 FC_SEM_DOMAINS 环境变量（按域分片）。幂等。"""
P = "/home1/lh/FreeChunker/train and preprocess/7_chunk_semantic.py"
s = open(P, encoding="utf-8").read()

MARK = "# fc-patch: domain sharding"
OLD = "selected_domains = ['single-doc','multi-doc','code-repo','long-dialogue','long-icl','long-structured']"
NEW = (
    MARK + "\n"
    "selected_domains = [d for d in os.environ.get(\n"
    "    'FC_SEM_DOMAINS',\n"
    "    'single-doc,multi-doc,code-repo,long-dialogue,long-icl,long-structured'\n"
    ").split(',') if d]"
)

if MARK in s:
    print("ALREADY_PATCHED")
elif OLD in s:
    s = s.replace(OLD, NEW)
    open(P, "w", encoding="utf-8").write(s)
    print("SEM_SHARD_PATCHED")
else:
    i = s.find("selected_domains")
    print("NOT_FOUND", repr(s[i:i + 120]))
