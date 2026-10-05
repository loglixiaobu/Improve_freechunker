from typing import List
import os

# 原脚本把默认 tokenizer 硬编码成作者集群上的绝对路径
#   /share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B
# 在别的机器上一定不存在，会让 Sentenceizer 初始化直接失败。
# 而它决定了「句子」的切分粒度（chunk_size=256 token），直接影响所有结果，
# 所以不能随便换成一个别的 tokenizer。
try:
    from src.paths import QWEN3_8B_LOCAL as _QWEN3_LOCAL
except Exception:  # 允许在没有 src 包的场景下单独使用本模块
    _QWEN3_LOCAL = os.environ.get("FC_QWEN3_LOCAL", "")


def setup_tokenizer(model_name=None):
    """Setup tokenizer

    按优先级尝试：显式传入 -> 项目内 models/Qwen3-8B -> hub 上的 Qwen/Qwen3-8B。
    全部失败才抛错，并把每个候选的失败原因带出来。
    """
    from transformers import AutoTokenizer

    candidates = []
    if model_name:
        candidates.append(model_name)
    if _QWEN3_LOCAL and os.path.isdir(_QWEN3_LOCAL):
        candidates.append(_QWEN3_LOCAL)
    candidates.append("Qwen/Qwen3-8B")

    errors = []
    for c in candidates:
        try:
            return AutoTokenizer.from_pretrained(c)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{c}: {type(e).__name__}: {e}")
    raise RuntimeError(
        "无法加载 Qwen3-8B tokenizer（Sentenizer 依赖它来切分句子）。已尝试：\n  "
        + "\n  ".join(errors)
    )

def fixed_size_chunking(text: str, tokenizer=None, chunk_size: int = 256, overlap: int = 0) -> List[str]:
    """
    Fixed-size chunking based on token count (Strict truncation)
    
    Args:
        text: Text to chunk
        tokenizer: Tokenizer
        chunk_size: Token count per chunk
        overlap: Overlapping token count
    """
    if tokenizer is None:
        tokenizer = setup_tokenizer()
    
    # Encode the entire text, do not add special tokens to keep it clean
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)
    
    chunks = []
    
    # Calculate step size
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # Prevent infinite loop, theoretically overlap should be smaller than chunk_size
    
    for i in range(0, total_tokens, step):
        # Truncate tokens for current chunk
        chunk_tokens = tokens[i : i + chunk_size]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
            
    return chunks

def traditional_chunking(text, tokenizer=None, chunk_size=256, overlap=0):
    """
    Fixed-size chunking based on tokens
    
    Args:
        text: Text to chunk
        tokenizer: Tokenizer
        chunk_size: Token count per chunk
        overlap: Overlapping token count
    """
    return fixed_size_chunking(text, tokenizer, chunk_size, overlap)

class TraditionalChunking:
    def __init__(self, model_name_or_path=None, tokenizer=None, chunk_size=256, overlap=0):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name_or_path is not None:
            self.tokenizer = setup_tokenizer(model_name_or_path)
        else:
            self.tokenizer = setup_tokenizer()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):
        return traditional_chunking(text, self.tokenizer, self.chunk_size, self.overlap)
