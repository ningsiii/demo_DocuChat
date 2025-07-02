from __future__ import annotations
from typing import List

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List
import numpy as np






# -------- 内部：挑选 Embedding 模型 ----------
def _get_embedding_model(provider: str, api_key) -> Embeddings:
    """
    根据 provider 返回对应的 Embeddings 实例
    """
    if provider == "openai":
        embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    elif provider == "dashscope":
        embeddings = DashScopeEmbeddings(dashscope_api_key=api_key, model="text-embedding-v2")
    elif provider == "deepseek":
        embeddings = BaseChatOpenAI(api_key=api_key, model='deepseek-chat')
    else:
        raise ValueError(f"不支持的嵌入提供商: {provider}")
    return embeddings

def _normalise(text: str) -> str:
    """最简单的清洗：去页眉页脚、连续空格 → 单空格"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines)


# -------- 对外 1：构建向量库 ----------
def build_vector_store(
        docs_clean: List[dict],
        provider: str,
        api_key: str
) -> FAISS:
    """
    docs_clean 每条至少含 "content"
    """
    embed_model = _get_embedding_model(provider, api_key)

    documents = [
        Document(page_content=_normalise(d["content"]), metadata=d)
        for d in docs_clean
    ]
    return FAISS.from_documents(documents, embed_model)


# -------- 对外 2：相似度检索 ----------
def top_k_context(
        vec_store: FAISS,
        query: str,
        k: int = 3
) -> List[str]:
    """
    返回与 query 最相似的 k 段文本（已去重）
    """
    hits = vec_store.similarity_search(query, k)
    # 去重
    seen = set()
    results = []
    for h in hits:
        txt = h.page_content
        if txt not in seen:
            seen.add(txt)
            results.append(txt)
    return results