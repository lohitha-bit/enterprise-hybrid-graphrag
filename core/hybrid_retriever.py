"""Hybrid Qdrant retrieval with Cohere Rerank v3."""

from __future__ import annotations

import logging
import os
from typing import Any

import cohere
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from core.ingestion import (
    DENSE_MODEL,
    DENSE_VECTOR_NAME,
    SPARSE_MODEL,
    SPARSE_VECTOR_NAME,
)

logger = logging.getLogger(__name__)

RERANK_MODEL = "rerank-english-v3.0"
HYBRID_CANDIDATE_LIMIT = 10
RERANK_TOP_N = 3
PREFETCH_LIMIT = 20


class RetrievedChunk(BaseModel):
    """A reranked retrieval hit with confidence score."""

    text: str = Field(..., min_length=1)
    score: float = Field(
        ...,
        description="Cohere relevance / confidence score (higher is better)",
    )
    rank: int = Field(..., ge=1, description="1-based rank after reranking")
    metadata: dict[str, Any] = Field(default_factory=dict)
    hybrid_score: float | None = Field(
        default=None,
        description="Qdrant RRF fusion score before Cohere rerank",
    )
    point_id: str | int | None = None


class HybridRerankRetriever:
    """
    Hybrid dense + BM25 retrieval from Qdrant, then Cohere Rerank v3.

    Flow:
      1. Prefetch dense (BAAI/bge-small-en-v1.5) and sparse (Qdrant/bm25) hits
      2. Fuse with Reciprocal Rank Fusion → top 10 candidates
      3. Rerank with ``rerank-english-v3.0`` → return top 3 + confidence scores
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        *,
        cohere_api_key: str | None = None,
        rerank_model: str = RERANK_MODEL,
        hybrid_limit: int = HYBRID_CANDIDATE_LIMIT,
        top_n: int = RERANK_TOP_N,
        prefetch_limit: int = PREFETCH_LIMIT,
    ) -> None:
        if hybrid_limit < 1:
            raise ValueError("hybrid_limit must be >= 1")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if top_n > hybrid_limit:
            raise ValueError("top_n cannot exceed hybrid_limit")

        self.client = client
        self.collection_name = collection_name
        self.rerank_model = rerank_model
        self.hybrid_limit = hybrid_limit
        self.top_n = top_n
        self.prefetch_limit = prefetch_limit

        api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key required: pass cohere_api_key=... or set COHERE_API_KEY"
            )
        self._cohere = cohere.Client(api_key)
        logger.info(
            "HybridRerankRetriever ready (collection=%s, rerank=%s, hybrid=%d→%d)",
            collection_name,
            rerank_model,
            hybrid_limit,
            top_n,
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Hybrid search + Cohere rerank.

        Returns the top ``top_n`` chunks (default 3) with confidence scores.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        query_text = query.strip()
        candidates = self._hybrid_search(query_text)
        if not candidates:
            logger.warning("No hybrid hits for query in '%s'", self.collection_name)
            return []

        return self._rerank(query_text, candidates)

    def _hybrid_search(self, query: str) -> list[dict[str, Any]]:
        """Prefetch dense + sparse, fuse with RRF, return top ``hybrid_limit`` hits."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query, model=DENSE_MODEL),
                    using=DENSE_VECTOR_NAME,
                    limit=self.prefetch_limit,
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=SPARSE_MODEL),
                    using=SPARSE_VECTOR_NAME,
                    limit=self.prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.hybrid_limit,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.get("text") or payload.get("page_content") or ""
            if not str(text).strip():
                continue
            metadata = {
                k: v for k, v in payload.items() if k not in {"text", "page_content"}
            }
            hits.append(
                {
                    "text": str(text),
                    "metadata": metadata,
                    "hybrid_score": float(point.score) if point.score is not None else None,
                    "point_id": point.id,
                }
            )

        logger.info(
            "Hybrid search returned %d/%d candidates for collection '%s'",
            len(hits),
            self.hybrid_limit,
            self.collection_name,
        )
        return hits

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[RetrievedChunk]:
        """Pass candidates to Cohere rerank-english-v3.0; keep top ``top_n``."""
        documents = [c["text"] for c in candidates]
        response = self._cohere.rerank(
            model=self.rerank_model,
            query=query,
            documents=documents,
            top_n=min(self.top_n, len(documents)),
            return_documents=False,
        )

        results: list[RetrievedChunk] = []
        for rank, item in enumerate(response.results, start=1):
            src = candidates[item.index]
            results.append(
                RetrievedChunk(
                    text=src["text"],
                    score=float(item.relevance_score),
                    rank=rank,
                    metadata=src.get("metadata") or {},
                    hybrid_score=src.get("hybrid_score"),
                    point_id=src.get("point_id"),
                )
            )

        logger.info(
            "Reranked to top %d (model=%s); best_score=%.4f",
            len(results),
            self.rerank_model,
            results[0].score if results else 0.0,
        )
        return results
