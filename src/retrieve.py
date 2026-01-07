"""
retrieve.py
Evidence retrieval system using Pathway for efficient chunk search
"""

import logging
from typing import List, Tuple
import numpy as np

from .chunk import Chunk, ChunkIndex
from .claims import Claim
from config import Config

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """
    Retrieves relevant narrative evidence for each backstory claim.
    Uses Pathway-integrated indexing for efficient search.
    """
    
    def __init__(self, chunk_index: ChunkIndex):
        """
        Initialize evidence retriever.
        
        Args:
            chunk_index: Index of narrative chunks
        """
        self.chunk_index = chunk_index
        self.chunks = chunk_index.get_chunks_in_order()
        
        logger.info(f"EvidenceRetriever initialized with {len(self.chunks)} chunks")
    
    def retrieve_evidence(self, claim: Claim, top_k: int = None) -> List[Chunk]:
        """
        Retrieve narrative chunks most relevant to a claim.
        
        Args:
            claim: Claim to find evidence for
            top_k: Number of chunks to retrieve (default: Config.TOP_K_CHUNKS)
            
        Returns:
            List of most relevant chunks in temporal order
        """
        top_k = top_k or Config.TOP_K_CHUNKS
        
        logger.debug(f"Retrieving evidence for: {claim.text[:50]}...")
        
        # Score all chunks for relevance
        scored_chunks = self._score_chunks(claim)
        
        # Take top-k by score
        top_chunks = scored_chunks[:top_k]
        
        # Re-sort by temporal order for consistent reasoning
        top_chunks.sort(key=lambda x: x[1].temporal_order)
        
        chunks = [chunk for score, chunk in top_chunks]
        
        logger.debug(f"Retrieved {len(chunks)} chunks for claim")
        return chunks
    
    def _score_chunks(self, claim: Claim) -> List[Tuple[float, Chunk]]:
        """
        Score all chunks for relevance to claim.
        
        Args:
            claim: Claim to score against
            
        Returns:
            List of (score, chunk) tuples sorted by score descending
        """
        claim_terms = self._extract_key_terms(claim.text)
        
        scored = []
        for chunk in self.chunks:
            score = self._compute_relevance_score(claim_terms, chunk)
            scored.append((score, chunk))
        
        # Sort by score descending
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return scored
    
    def _extract_key_terms(self, text: str) -> set[str]:
        """
        Extract key terms from claim text for matching.
        
        Args:
            text: Claim text
            
        Returns:
            Set of lowercase key terms
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Split and filter
        terms = set(text.split()) - stop_words
        
        # Keep terms that are at least 3 characters
        terms = {t for t in terms if len(t) >= 3}
        
        return terms
    
    def _compute_relevance_score(self, query_terms: set[str], chunk: Chunk) -> float:
        """
        Compute relevance score between query terms and chunk.
        
        Args:
            query_terms: Set of terms from claim
            chunk: Chunk to score
            
        Returns:
            Relevance score (higher = more relevant)
        """
        chunk_text = chunk.text.lower()
        chunk_terms = set(chunk_text.split())
        
        # Term overlap score
        overlap = len(query_terms & chunk_terms)
        
        if overlap == 0:
            return 0.0
        
        # Base score from overlap
        score = overlap / len(query_terms)
        
        # Boost for term proximity (terms appearing close together)
        proximity_boost = self._compute_proximity_bonus(
            query_terms, chunk_text
        )
        score += proximity_boost
        
        # Boost for multiple occurrences
        frequency_boost = 0.0
        for term in query_terms:
            count = chunk_text.count(term)
            if count > 1:
                frequency_boost += 0.1 * (count - 1)
        
        score += min(frequency_boost, 0.5)  # Cap frequency boost
        
        return score
    
    def _compute_proximity_bonus(self, terms: set[str], text: str) -> float:
        """
        Compute bonus for terms appearing close together.
        
        Args:
            terms: Query terms
            text: Chunk text
            
        Returns:
            Proximity bonus score
        """
        words = text.split()
        term_positions = {term: [] for term in terms}
        
        # Find all positions of each term
        for i, word in enumerate(words):
            for term in terms:
                if term in word:
                    term_positions[term].append(i)
        
        # If not all terms found, no bonus
        if any(not positions for positions in term_positions.values()):
            return 0.0
        
        # Find minimum distance between any pair of terms
        min_distance = float('inf')
        term_list = list(terms)
        
        for i in range(len(term_list)):
            for j in range(i + 1, len(term_list)):
                term1_pos = term_positions[term_list[i]]
                term2_pos = term_positions[term_list[j]]
                
                for pos1 in term1_pos:
                    for pos2 in term2_pos:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)
        
        # Convert distance to bonus (closer = higher bonus)
        if min_distance < 10:
            return 0.5
        elif min_distance < 50:
            return 0.3
        elif min_distance < 100:
            return 0.1
        
        return 0.0
    
    def retrieve_temporal_context(self, chunks: List[Chunk], 
                                  window: int = 1) -> List[Chunk]:
        """
        Expand chunks to include temporal neighbors.
        
        Args:
            chunks: Initial set of chunks
            window: Number of chunks before/after to include
            
        Returns:
            Expanded list with temporal context
        """
        expanded = set()
        
        for chunk in chunks:
            # Add the chunk itself
            expanded.add(chunk)
            
            # Add neighbors
            for offset in range(-window, window + 1):
                neighbor_order = chunk.temporal_order + offset
                try:
                    neighbor = self.chunk_index.get_chunk_by_order(neighbor_order)
                    expanded.add(neighbor)
                except IndexError:
                    continue  # Out of bounds
        
        # Sort by temporal order
        result = sorted(expanded, key=lambda c: c.temporal_order)
        
        return result


class EvidenceAggregator:
    """
    Aggregates evidence from multiple chunks for claim verification.
    """
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize aggregator.
        
        Args:
            max_tokens: Maximum tokens to include in aggregated evidence
        """
        self.max_tokens = max_tokens
        
    def aggregate_evidence(self, chunks: List[Chunk]) -> str:
        """
        Combine chunks into coherent evidence text.
        
        Args:
            chunks: List of chunks in temporal order
            
        Returns:
            Aggregated evidence text
        """
        if not chunks:
            return ""
        
        # Simple concatenation with separators
        evidence_parts = []
        total_words = 0
        
        for chunk in chunks:
            # Approximate tokens (1 token ≈ 0.75 words)
            chunk_tokens = int(chunk.word_count * 0.75)
            
            if total_words + chunk_tokens > self.max_tokens:
                break  # Stop if we exceed limit
            
            evidence_parts.append(
                f"[Chunk {chunk.temporal_order}]\n{chunk.text}\n"
            )
            total_words += chunk_tokens
        
        return "\n---\n\n".join(evidence_parts)
    
    def create_evidence_summary(self, chunks: List[Chunk]) -> str:
        """
        Create a summary of evidence sources.
        
        Args:
            chunks: Chunks used as evidence
            
        Returns:
            Summary string
        """
        if not chunks:
            return "No evidence found"
        
        chunk_ids = [c.chunk_id for c in chunks]
        positions = [c.temporal_order for c in chunks]
        
        summary = (
            f"Evidence from {len(chunks)} chunks "
            f"(positions: {min(positions)}-{max(positions)} in narrative)"
        )
        
        return summary


class PathwayEvidenceIndex:
    """
    Wrapper for Pathway-based evidence indexing.
    In production, this would use Pathway's vector store.
    For hackathon, we use optimized in-memory structures.
    """
    
    def __init__(self, chunk_index: ChunkIndex):
        """
        Initialize Pathway evidence index.
        
        Args:
            chunk_index: Chunk index to wrap
        """
        self.chunk_index = chunk_index
        self.chunks = chunk_index.get_chunks_in_order()
        
        # Build inverted index for fast lookup
        self.inverted_index = self._build_inverted_index()
        
        logger.info("PathwayEvidenceIndex initialized")
    
    def _build_inverted_index(self) -> dict[str, set[int]]:
        """
        Build inverted index mapping terms to chunk orders.
        
        Returns:
            Dict mapping term -> set of chunk temporal orders
        """
        index = {}
        
        for chunk in self.chunks:
            terms = set(chunk.text.lower().split())
            
            for term in terms:
                if len(term) >= 3:  # Only index meaningful terms
                    if term not in index:
                        index[term] = set()
                    index[term].add(chunk.temporal_order)
        
        logger.info(f"Built inverted index with {len(index)} terms")
        return index
    
    def fast_retrieve(self, query_terms: set[str], top_k: int = 5) -> List[Chunk]:
        """
        Fast retrieval using inverted index.
        
        Args:
            query_terms: Set of terms to search for
            top_k: Number of chunks to return
            
        Returns:
            Top-k most relevant chunks
        """
        # Find chunks containing any query term
        candidate_orders = set()
        
        for term in query_terms:
            term = term.lower()
            if term in self.inverted_index:
                candidate_orders.update(self.inverted_index[term])
        
        if not candidate_orders:
            return []
        
        # Score candidates
        candidates = [self.chunk_index.get_chunk_by_order(order) 
                     for order in candidate_orders]
        
        # Simple scoring based on term overlap
        scored = []
        for chunk in candidates:
            chunk_terms = set(chunk.text.lower().split())
            overlap = len(query_terms & chunk_terms)
            scored.append((overlap, chunk))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return [chunk for score, chunk in scored[:top_k]]