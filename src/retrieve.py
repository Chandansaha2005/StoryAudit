"""
retrieve.py
Evidence retrieval helpers and a simple Pathway-compatible index.
"""

import logging
from typing import List, Tuple, Set

from .chunk import Chunk, ChunkIndex
from .claims import Claim
from config import Config

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Find relevant evidence for claims."""

    def __init__(self, chunk_index: ChunkIndex):
        self.chunk_index = chunk_index
        self.chunks = chunk_index.get_chunks_in_order()
        logger.info(f"EvidenceRetriever initialized with {len(self.chunks)} chunks")

    def retrieve_evidence(self, claim: Claim, top_k: int | None = None) -> List[Chunk]:
        top_k = top_k or Config.TOP_K_CHUNKS
        terms = self._extract_key_terms(claim.text)

        scored = []
        for chunk in self.chunks:
            score = self._compute_relevance_score(terms, chunk)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [chunk for _, chunk in scored[:top_k]]
        # ensure temporal order
        top_chunks.sort(key=lambda c: c.temporal_order)
        return top_chunks

    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text."""
        stop_words = {"the", "and", "a", "an", "of", "in", "on", "to", "is", "was", "were", "it"}
        terms = {t.lower().strip(".,;:()[]") for t in text.split()}
        terms = {t for t in terms if len(t) >= 3 and t not in stop_words}
        return terms

    def _compute_relevance_score(self, query_terms: Set[str], chunk: Chunk) -> float:
        chunk_text = chunk.text.lower()
        chunk_terms = set(chunk_text.split())
        overlap = len(query_terms & chunk_terms)
        if overlap == 0:
            return 0.0

        score = overlap / max(1, len(query_terms))
        # small proximity bonus
        score += self._compute_proximity_bonus(query_terms, chunk_text)
        return score

    def _compute_proximity_bonus(self, terms: Set[str], text: str) -> float:
        words = text.split()
        positions = []
        for i, w in enumerate(words):
            for t in terms:
                if t in w:
                    positions.append(i)

        if len(positions) < 2:
            return 0.0

        min_dist = min(abs(a - b) for i, a in enumerate(positions) for b in positions[i + 1 :])
        if min_dist < 10:
            return 0.3
        if min_dist < 50:
            return 0.1
        return 0.0


class EvidenceAggregator:
    """Combine chunks into evidence text."""

    def __init__(self, max_tokens: int = 4000) -> None:
        self.max_tokens = max_tokens

    def aggregate_evidence(self, chunks: List[Chunk]) -> str:
        if not chunks:
            return ""

        parts = []
        total_tokens = 0
        for c in chunks:
            approx_tokens = int(c.word_count * 0.75)
            if total_tokens + approx_tokens > self.max_tokens:
                break
            parts.append(f"[Chunk {c.temporal_order}]\n{c.text}\n")
            total_tokens += approx_tokens

        return "\n---\n\n".join(parts)


class PathwayEvidenceIndex:
    """Inverted index for chunk retrieval."""

    def __init__(self, chunk_index: ChunkIndex):
        self.chunk_index = chunk_index
        self.chunks = chunk_index.get_chunks_in_order()
        self.inverted_index = self._build_inverted_index()

    def _build_inverted_index(self) -> dict:
        index: dict = {}
        for chunk in self.chunks:
            for term in set(chunk.text.lower().split()):
                if len(term) < 3:
                    continue
                index.setdefault(term, set()).add(chunk.temporal_order)
        logger.info(f"Built inverted index with {len(index)} terms")
        return index

    def fast_retrieve(self, query_terms: Set[str], top_k: int = 5) -> List[Chunk]:
        candidate_orders = set()
        for t in query_terms:
            candidate_orders.update(self.inverted_index.get(t, set()))

        candidates = [self.chunk_index.get_chunk_by_order(o) for o in candidate_orders]
        scored = [(len(query_terms & set(c.text.lower().split())), c) for c in candidates]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:top_k]]
