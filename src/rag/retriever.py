"""
知识库检索：markdown 按二级标题切块，向量检索（bge-small-zh）+ BM25 混合召回，RRF 融合排序。

语料只有几十个 chunk，用 numpy 暴力计算余弦相似度即可，不需要引入向量数据库。
"""
import json
import re
from functools import lru_cache

import jieba
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from config import EMBED_CACHE, EMBED_MODEL, KB_DIR, RAG_DIR

jieba.setLogLevel("WARNING")

CHUNKS_PATH = RAG_DIR / "chunks.json"
EMB_PATH = RAG_DIR / "embeddings.npy"
RRF_K = 60


def split_markdown(doc_name: str, text: str) -> list[dict]:
    """按 `## ` 切块，每块带上文档标题，保证 chunk 脱离上下文也能看懂。"""
    title_match = re.match(r"#\s+(.+)", text)
    title = title_match.group(1).strip() if title_match else doc_name
    parts = re.split(r"\n(?=## )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part.startswith("## "):
            # 文档开头的导语部分
            body = re.sub(r"^#\s+.+\n?", "", part).strip()
            if body:
                chunks.append({"doc": doc_name, "section": "概述",
                               "text": f"【{title}】\n{body}"})
            continue
        section = part.splitlines()[0][3:].strip()
        chunks.append({"doc": doc_name, "section": section,
                       "text": f"【{title} / {section}】\n{part}"})
    return chunks


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip() and not re.fullmatch(r"\W+", t)]


@lru_cache(maxsize=1)
def _embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL, cache_dir=str(EMBED_CACHE))


def _embed(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(_embedder().embed(texts)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def build_index() -> int:
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        chunks += split_markdown(path.stem, path.read_text(encoding="utf-8"))
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(EMB_PATH, _embed([c["text"] for c in chunks]))
    return len(chunks)


class Retriever:
    def __init__(self):
        if not CHUNKS_PATH.exists():
            build_index()
        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.emb = np.load(EMB_PATH)
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        # bge 检索时给 query 加官方推荐的指令前缀
        q = _embed(["为这个句子生成表示以用于检索相关文章：" + query])[0]
        dense_rank = np.argsort(-(self.emb @ q))
        sparse_rank = np.argsort(-self.bm25.get_scores(_tokenize(query)))

        scores = np.zeros(len(self.chunks))
        for rank_list in (dense_rank, sparse_rank):
            for r, idx in enumerate(rank_list):
                scores[idx] += 1.0 / (RRF_K + r + 1)
        top = np.argsort(-scores)[:top_k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 4)} for i in top]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


if __name__ == "__main__":
    print(f"indexed {build_index()} chunks")
    r = Retriever()
    for q in ["坏账率怎么算", "年轻低额度客群的定义", "education=5 是什么意思",
              "坏账率上升怎么分析原因", "SHAP 值怎么解读", "合作方甩单"]:
        hits = r.search(q, top_k=2)
        print(q, "->", [(h["doc"], h["section"]) for h in hits])
