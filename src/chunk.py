"""
chunk.py
Intelligent chunking with temporal ordering for long narratives
"""

import logging
import re
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Narrative chunk with metadata."""
    chunk_id: str
    text: str
    start_pos: int  # Character position in original text
    end_pos: int
    word_count: int
    temporal_order: int  # Sequential order in narrative (0, 1, 2, ...)
    
    def __repr__(self):
        return f"Chunk(id={self.chunk_id}, words={self.word_count}, order={self.temporal_order})"


class NarrativeChunker:
    """Split narratives maintaining temporal order."""
    
    def __init__(self, chunk_size: int = 2500, overlap: int = 300):
        """Initialize chunker."""
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        
        logger.info(f"Chunker initialized: size={chunk_size}, overlap={overlap}")
    
    def chunk_narrative(self, text: str, story_id: str) -> List[Chunk]:
        """Split narrative into overlapping chunks."""
        logger.info("Starting narrative chunking...")
        
        # try to detect chapter boundaries
        chapters = self._detect_chapters(text)
        
        if chapters:
            logger.info(f"Detected {len(chapters)} chapters")
            chunks = self._chunk_by_chapters(chapters, story_id)
        else:
            logger.info("No clear chapters detected, using sliding window")
            chunks = self._chunk_sliding_window(text, story_id)
        
        logger.info(f"Created {len(chunks)} chunks")
        self._log_chunk_statistics(chunks)
        
        return chunks
    
    def _detect_chapters(self, text: str) -> List[Tuple[str, str]]:
        """Try to detect chapter boundaries."""
        # common chapter patterns
        patterns = [
            r'\n\s*CHAPTER\s+[IVXLCDM\d]+\s*[:\-]?\s*[^\n]*\n',
            r'\n\s*Chapter\s+[IVXLCDM\d]+\s*[:\-]?\s*[^\n]*\n',
            r'\n\s*[IVXLCDM]+\.\s*[^\n]*\n',
            r'\n\s*\d+\.\s*[^\n]*\n',
        ]
    
    def _chunk_by_chapters(self, chapters: List[Tuple[str, str]], 
                          story_id: str) -> List[Chunk]:
        """Split chapters into chunks."""
        chunks = []
        chunk_counter = 0
        char_position = 0
        
        current_batch_text = ""
        current_batch_start = 0
        
        for chapter_title, chapter_text in chapters:
            chapter_words = chapter_text.split()
            
            # If chapter is small, accumulate
            if len(chapter_words) < self.chunk_size * 0.5:
                if not current_batch_text:
                    current_batch_start = char_position
                current_batch_text += f"\n\n{chapter_title}\n{chapter_text}"
            else:
                # Flush accumulated small chapters first
                if current_batch_text:
                    chunk = self._create_chunk(
                        current_batch_text, 
                        story_id, 
                        chunk_counter, 
                        current_batch_start
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                    current_batch_text = ""
                
                # Split large chapter into sub-chunks
                sub_chunks = self._split_large_text(
                    chapter_text, 
                    story_id, 
                    chunk_counter, 
                    char_position
                )
                chunks.extend(sub_chunks)
                chunk_counter += len(sub_chunks)
            
            char_position += len(chapter_title) + len(chapter_text) + 2
        
        # Flush remaining
        if current_batch_text:
            chunk = self._create_chunk(
                current_batch_text, 
                story_id, 
                chunk_counter, 
                current_batch_start
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_sliding_window(self, text: str, story_id: str) -> List[Chunk]:
        """Chunk text using sliding window."""
        words = text.split()
        chunks = []
        
        start_word = 0
        chunk_counter = 0
        
        while start_word < len(words):
            end_word = min(start_word + self.chunk_size, len(words))
            chunk_words = words[start_word:end_word]
            
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character position
            char_start = len(' '.join(words[:start_word]))
            
            chunk = self._create_chunk(
                chunk_text, 
                story_id, 
                chunk_counter, 
                char_start
            )
            chunks.append(chunk)
            
            # Move window forward (with overlap)
            start_word += self.chunk_size - self.overlap
            chunk_counter += 1
            
            # Prevent infinite loop
            if end_word == len(words):
                break
        
        return chunks
    
    def _split_large_text(self, text: str, story_id: str, 
                         start_counter: int, start_pos: int) -> List[Chunk]:
        """Split large text into sub-chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            end = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end]
            
            chunk_text = ' '.join(chunk_words)
            char_offset = len(' '.join(words[:i]))
            
            chunk = self._create_chunk(
                chunk_text,
                story_id,
                start_counter + len(chunks),
                start_pos + char_offset
            )
            chunks.append(chunk)
            
            if end == len(words):
                break
        
        return chunks
    
    def _create_chunk(self, text: str, story_id: str, 
                     order: int, start_pos: int) -> Chunk:
        """Create Chunk object."""
        word_count = len(text.split())
        
        return Chunk(
            chunk_id=f"{story_id}_chunk_{order:03d}",
            text=text.strip(),
            start_pos=start_pos,
            end_pos=start_pos + len(text),
            word_count=word_count,
            temporal_order=order
        )
    
    def _log_chunk_statistics(self, chunks: List[Chunk]):
        """Log chunk statistics."""
        if not chunks:
            return
        
        word_counts = [c.word_count for c in chunks]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        
        logger.info(f"Chunk statistics: avg={avg_words:.0f}, "
                   f"min={min_words}, max={max_words}")
    
    def get_temporal_context(self, chunk: Chunk, all_chunks: List[Chunk], 
                           window: int = 1) -> List[Chunk]:
        """Get adjacent chunks for context."""
        idx = chunk.temporal_order
        start = max(0, idx - window)
        end = min(len(all_chunks), idx + window + 1)
        
        return all_chunks[start:end]


class ChunkIndex:
    """Index for efficient chunk lookup."""
    
    def __init__(self, chunks: List[Chunk]):
        """
        Initialize index from chunks.
        
        Args:
            chunks: List of chunks in temporal order
        """
        self.chunks = sorted(chunks, key=lambda c: c.temporal_order)
        self.id_to_chunk = {c.chunk_id: c for c in chunks}
        
        logger.info(f"Chunk index built with {len(chunks)} chunks")
    
    def get_chunk(self, chunk_id: str) -> Chunk:
        """Get chunk by ID."""
        if chunk_id not in self.id_to_chunk:
            raise KeyError(f"Chunk {chunk_id} not found")
        return self.id_to_chunk[chunk_id]
    
    def get_chunks_in_order(self) -> List[Chunk]:
        """Get all chunks in temporal order."""
        return self.chunks
    
    def get_chunk_by_order(self, order: int) -> Chunk:
        """Get chunk by temporal order index."""
        if 0 <= order < len(self.chunks):
            return self.chunks[order]
        raise IndexError(f"Order {order} out of range [0, {len(self.chunks)})")
    
    def search_chunks_by_text(self, query: str, top_k: int = 5) -> List[Chunk]:
        """
        Simple keyword-based chunk search.
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            
        Returns:
            List of most relevant chunks
        """
        query_words = set(query.lower().split())
        
        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words & chunk_words)
            
            # Boost score if query words appear close together
            score = overlap
            if overlap > 0:
                # Simple positional scoring
                for word in query_words:
                    if word in chunk.text.lower():
                        score += 0.5
            
            scored.append((score, chunk))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in scored[:top_k] if score > 0]