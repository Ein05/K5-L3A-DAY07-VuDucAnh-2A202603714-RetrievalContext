"""Retrieval methods for Day 7 Agentic RAG: Lexical (TF-IDF), Dense (FastEmbed 384d), and Hybrid (RRF)."""
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Any
import numpy as np
from fastembed import TextEmbedding


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text with NFC normalization."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


class TfidfRetriever:
    """Phương pháp 1: Lexical Retrieval bằng TF-IDF."""
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counter: Counter) -> dict[str, float]:
        norm = math.sqrt(sum((c * self.idf.get(t, 0)) ** 2 for t, c in counter.items())) or 1.0
        return {term: (count * self.idf.get(term, 0)) / norm for term, count in counter.items()}

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_vec = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if filters:
                match = all(chunk.get("metadata", {}).get(k) == v for k, v in filters.items())
                if not match:
                    continue
            score = sum(val * vector.get(term, 0) for term, val in query_vec.items())
            if score > 0:
                results.append({**chunk, "score": float(score), "strategy": "tfidf"})
        results.sort(key=lambda c: (-c["score"], c["id"]))
        return results[:top_k]


class DenseRetriever:
    """Phương pháp 2: Dense Retrieval kế thừa từ Day 6 (FastEmbed paraphrase-multilingual-MiniLM-L12-v2, 384d)."""
    def __init__(self, chunks: list[dict[str, Any]], cache_dir: str | None = None):
        self.chunks = chunks
        self.embedder = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Cache embeddings to avoid re-computing on every run
        cache_path = os.path.join(cache_dir, "dense_embeddings.npy") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
        else:
            texts = [c["text"] for c in chunks]
            self.embeddings = np.array(list(self.embedder.embed(texts)), dtype=np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-9)
            if cache_path:
                np.save(cache_path, self.embeddings)

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_emb = np.array(list(self.embedder.embed([query]))[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm
        scores = np.dot(self.embeddings, q_emb)

        results = []
        for chunk, score in zip(self.chunks, scores):
            if filters:
                match = all(chunk.get("metadata", {}).get(k) == v for k, v in filters.items())
                if not match:
                    continue
            results.append({**chunk, "score": float(score), "strategy": "dense"})
        results.sort(key=lambda c: (-c["score"], c["id"]))
        return results[:top_k]


class HybridRetriever:
    """Phương pháp 3: Hybrid Retrieval kết hợp Dense + Lexical thông qua Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, tfidf_retriever: TfidfRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.tfidf = tfidf_retriever
        self.rrf_k = rrf_k
        self.chunks_by_id = {c["id"]: c for c in dense_retriever.chunks}

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        # Retrieve candidate pools from both
        cand_dense = self.dense.search(query, filters=filters, top_k=20)
        cand_tfidf = self.tfidf.search(query, filters=filters, top_k=20)

        rrf_scores: dict[str, float] = {}
        for rank, item in enumerate(cand_dense, start=1):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, item in enumerate(cand_tfidf, start=1):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (self.rrf_k + rank)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        results = []
        for cid in sorted_ids[:top_k]:
            base = self.chunks_by_id[cid]
            results.append({
                **base,
                "score": float(rrf_scores[cid]),
                "strategy": "hybrid_rrf"
            })
        return results
