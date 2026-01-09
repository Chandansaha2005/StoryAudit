"""
Advanced chunking strategies for long documents.
Provides hierarchical chunking, sliding windows, and context preservation.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SmartChunker:
    """Smart document chunking with multiple strategies."""
    
    def __init__(self, chunk_size: int = 2500, overlap: int = 300, min_chunk_size: int = 100):
        """
        Initialize smart chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        logger.info(f"SmartChunker initialized: size={chunk_size}, overlap={overlap}")
    
    def chunk_by_sentences(self, text: str) -> List[Dict]:
        """
        Chunk text by sentence boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dicts with 'text', 'start', 'end', 'sentence_count'
        """
        # Split into sentences (simple regex)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks = []
        current_chunk = []
        current_size = 0
        start_pos = 0
        
        for sent in sentences:
            sent_size = len(sent) + 1  # +1 for space
            
            # Start new chunk if current would exceed size
            if current_size + sent_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        'text': chunk_text,
                        'start': start_pos,
                        'end': start_pos + len(chunk_text),
                        'sentence_count': len(current_chunk),
                        'type': 'sentence_boundary'
                    })
                    start_pos += len(chunk_text) + 1
                
                current_chunk = [sent]
                current_size = sent_size
            else:
                current_chunk.append(sent)
                current_size += sent_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start': start_pos,
                    'end': start_pos + len(chunk_text),
                    'sentence_count': len(current_chunk),
                    'type': 'sentence_boundary'
                })
        
        logger.info(f"Created {len(chunks)} chunks by sentence boundaries")
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[Dict]:
        """
        Chunk text by paragraph boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dicts
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        start_pos = 0
        
        for para in paragraphs:
            para_size = len(para) + 2  # +2 for newlines
            
            # Start new chunk if current would exceed size
            if current_size + para_size > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        'text': chunk_text,
                        'start': start_pos,
                        'end': start_pos + len(chunk_text),
                        'paragraph_count': len(current_chunk),
                        'type': 'paragraph_boundary'
                    })
                    start_pos += len(chunk_text) + 2
                
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start': start_pos,
                    'end': start_pos + len(chunk_text),
                    'paragraph_count': len(current_chunk),
                    'type': 'paragraph_boundary'
                })
        
        logger.info(f"Created {len(chunks)} chunks by paragraph boundaries")
        return chunks
    
    def chunk_hierarchical(self, text: str) -> Dict:
        """
        Create hierarchical chunk structure: document -> sections -> paragraphs -> chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            Hierarchical structure dict
        """
        # Detect sections by numbered headings
        section_pattern = r'^\d+\.\s+(.+)$'
        lines = text.split('\n')
        
        sections = []
        current_section = {'title': 'Introduction', 'content': [], 'start': 0}
        
        for i, line in enumerate(lines):
            match = re.match(section_pattern, line)
            if match and i > 0:  # Don't split on first line
                # Save current section
                content = '\n'.join(current_section['content'])
                if content.strip():
                    current_section['text'] = content
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': match.group(1),
                    'content': [],
                    'start': i
                }
            else:
                current_section['content'].append(line)
        
        # Save final section
        content = '\n'.join(current_section['content'])
        if content.strip():
            current_section['text'] = content
            sections.append(current_section)
        
        # Chunk each section
        hierarchy = {
            'type': 'document',
            'sections': []
        }
        
        for section in sections:
            section_text = section.get('text', '')
            section_chunks = self.chunk_by_paragraphs(section_text)
            
            hierarchy['sections'].append({
                'title': section['title'],
                'chunks': section_chunks,
                'chunk_count': len(section_chunks)
            })
        
        logger.info(f"Created hierarchical structure with {len(sections)} sections")
        return hierarchy
    
    def chunk_with_sliding_window(self, text: str) -> List[Dict]:
        """
        Create chunks with sliding window and overlap for context preservation.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of overlapping chunks
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            end = min(i + self.chunk_size, len(text))
            chunk_text = text[i:end]
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start': i,
                    'end': end,
                    'size': len(chunk_text),
                    'type': 'sliding_window',
                    'overlap_start': i + (self.chunk_size - self.overlap),
                    'overlap_end': end - (self.chunk_size - self.overlap) if end < len(text) else end
                })
        
        logger.info(f"Created {len(chunks)} overlapping chunks with sliding window")
        return chunks
    
    def get_chunk_context(self, chunks: List[Dict], chunk_index: int, context_chunks: int = 1) -> str:
        """
        Get a chunk with surrounding context.
        
        Args:
            chunks: List of chunks
            chunk_index: Index of target chunk
            context_chunks: Number of surrounding chunks to include
            
        Returns:
            Chunk text with context
        """
        start_idx = max(0, chunk_index - context_chunks)
        end_idx = min(len(chunks), chunk_index + context_chunks + 1)
        
        context_parts = []
        for i in range(start_idx, end_idx):
            if i == chunk_index:
                context_parts.append(f"[TARGET]\n{chunks[i]['text']}\n[/TARGET]")
            else:
                context_parts.append(chunks[i]['text'])
        
        return '\n\n---\n\n'.join(context_parts)


class ChunkRelationshipTracker:
    """Track relationships between chunks for coherence."""
    
    def __init__(self):
        """Initialize tracker."""
        self.chunks: List[Dict] = []
        self.relationships: Dict = {}
        logger.info("ChunkRelationshipTracker initialized")
    
    def add_chunk(self, chunk: Dict):
        """Add chunk to tracker."""
        chunk_id = len(self.chunks)
        chunk['id'] = chunk_id
        self.chunks.append(chunk)
        self.relationships[chunk_id] = {
            'previous': chunk_id - 1 if chunk_id > 0 else None,
            'next': None
        }
        
        # Update previous chunk's next pointer
        if chunk_id > 0:
            self.relationships[chunk_id - 1]['next'] = chunk_id
    
    def get_chunk_sequence(self, start_id: int, length: int = 3) -> List[Dict]:
        """Get a sequence of chunks."""
        sequence = []
        current_id = start_id
        
        for _ in range(length):
            if current_id is not None and current_id < len(self.chunks):
                sequence.append(self.chunks[current_id])
                current_id = self.relationships[current_id]['next']
            else:
                break
        
        return sequence
    
    def get_chunk_neighborhood(self, chunk_id: int, radius: int = 1) -> List[Dict]:
        """Get chunks in neighborhood of target chunk."""
        neighborhood = []
        
        # Add previous chunks
        prev_id = chunk_id
        for _ in range(radius):
            if prev_id is not None and prev_id >= 0:
                neighborhood.insert(0, self.chunks[prev_id])
                prev_id = self.relationships[prev_id]['previous']
            else:
                break
        
        # Add current chunk
        if chunk_id < len(self.chunks):
            neighborhood.append(self.chunks[chunk_id])
        
        # Add next chunks
        next_id = self.relationships[chunk_id]['next']
        for _ in range(radius):
            if next_id is not None and next_id < len(self.chunks):
                neighborhood.append(self.chunks[next_id])
                next_id = self.relationships[next_id]['next']
            else:
                break
        
        return neighborhood
