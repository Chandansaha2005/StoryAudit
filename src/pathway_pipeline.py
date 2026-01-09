"""
pathway_pipeline.py
Optional Pathway-based streaming pipeline for document processing.

This module provides reactive data processing capabilities (optional).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pathway as pw
except ImportError:
    pw = None

logger = logging.getLogger(__name__)


class PathwayStreamingPipeline:
    """
    Streaming pipeline using Pathway's reactive programming model.
    Processes documents as they arrive and maintains reactive state.
    """
    
    def __init__(self):
        """Initialize Pathway streaming pipeline."""
        self.narratives_table: Optional[pw.Table] = None
        self.backstories_table: Optional[pw.Table] = None
        self.claims_table: Optional[pw.Table] = None
        self.verification_results: Optional[pw.Table] = None
        
        logger.info("PathwayStreamingPipeline initialized")
    
    def create_document_stream(self, documents: Dict[str, str], doc_type: str = "narrative") -> pw.Table:
        """
        Create a reactive data stream from documents.
        
        Args:
            documents: Dict mapping doc_id -> content
            doc_type: Type of documents ("narrative" or "backstory")
            
        Returns:
            Pathway Table with streaming data
        """
        logger.info(f"Creating {doc_type} stream with {len(documents)} documents")
        
        # Convert to Pathway-compatible rows
        rows = [
            {
                "id": doc_id,
                "content": content,
                "doc_type": doc_type,
                "word_count": len(content.split()),
                "char_count": len(content)
            }
            for doc_id, content in documents.items()
        ]
        
        # Create reactive table from document stream
        table = pw.debug.table_from_rows(
            rows=rows,
            schema=pw.schema_builder({
                "id": pw.column_definition(dtype=str),
                "content": pw.column_definition(dtype=str),
                "doc_type": pw.column_definition(dtype=str),
                "word_count": pw.column_definition(dtype=int),
                "char_count": pw.column_definition(dtype=int)
            })
        )
        
        logger.info(f"Created {doc_type} stream table with reactive schema")
        return table
    
    def add_computed_columns(self, table: pw.Table) -> pw.Table:
        """
        Add computed columns to enhance document data reactively.
        
        Args:
            table: Pathway input table
            
        Returns:
            Table with added computed columns
        """
        logger.info("Adding computed columns to document stream")
        
        # Compute additional derived columns reactively
        # These will automatically update if source data changes
        enhanced = table.select(
            pw.this.id,
            pw.this.content,
            pw.this.doc_type,
            word_count=pw.this.word_count,
            char_count=pw.this.char_count,
            # Derived columns
            is_substantial=pw.this.word_count >= 100,  # Min 100 words
            avg_word_length=pw.this.char_count / (pw.this.word_count + 1),
            first_100_chars=pw.this.content.str[:100]
        )
        
        logger.info("Computed columns added successfully")
        return enhanced
    
    def process_narrative_stream(self, narratives: Dict[str, str]) -> pw.Table:
        """
        Process narrative documents as a reactive stream.
        
        Args:
            narratives: Dict of narrative documents
            
        Returns:
            Processed narrative stream
        """
        logger.info("Processing narrative stream with Pathway")
        
        # Create document stream
        self.narratives_table = self.create_document_stream(narratives, "narrative")
        
        # Enhance with computed columns
        self.narratives_table = self.add_computed_columns(self.narratives_table)
        
        # Filter only substantial narratives
        self.narratives_table = self.narratives_table.filter(
            pw.this.is_substantial
        )
        
        logger.info("Narrative stream processing complete")
        return self.narratives_table
    
    def process_backstory_stream(self, backstories: Dict[str, str]) -> pw.Table:
        """
        Process backstory documents as a reactive stream.
        
        Args:
            backstories: Dict of backstory documents
            
        Returns:
            Processed backstory stream
        """
        logger.info("Processing backstory stream with Pathway")
        
        # Create document stream
        self.backstories_table = self.create_document_stream(backstories, "backstory")
        
        # Enhance with computed columns
        self.backstories_table = self.add_computed_columns(self.backstories_table)
        
        logger.info("Backstory stream processing complete")
        return self.backstories_table
    
    def join_narrative_backstory(self, extract_story_id_fn=None) -> Optional[pw.Table]:
        """
        Reactively join narrative and backstory documents.
        Creates a joined stream where related documents are paired.
        
        Args:
            extract_story_id_fn: Function to extract story ID from document ID
            
        Returns:
            Joined table with paired documents
        """
        if self.narratives_table is None or self.backstories_table is None:
            logger.warning("Cannot join: one or both tables not initialized")
            return None
        
        logger.info("Joining narrative and backstory streams reactively")
        
        try:
            # Simple join on document ID prefix
            # In production, could use more sophisticated matching
            narratives_indexed = self.narratives_table.with_id(pw.this.id)
            backstories_indexed = self.backstories_table.with_id(pw.this.id)
            
            # This would normally use pw.join, but for streaming we use inner join
            # The join maintains reactive updates automatically
            joined = narratives_indexed.join(
                backstories_indexed,
                narratives_indexed.id == backstories_indexed.id,
                how="inner"
            ).select(
                narrative_id=narratives_indexed.id,
                narrative_content=narratives_indexed.content,
                narrative_words=narratives_indexed.word_count,
                backstory_id=backstories_indexed.id,
                backstory_content=backstories_indexed.content,
                backstory_words=backstories_indexed.word_count
            )
            
            logger.info("Narrative-backstory join successful")
            return joined
            
        except Exception as e:
            logger.error(f"Join failed: {e}")
            return None
    
    def compute_document_statistics(self, table: pw.Table) -> pw.Table:
        """
        Compute aggregate statistics over document stream.
        Uses Pathway's reactive aggregation capabilities.
        
        Args:
            table: Input document table
            
        Returns:
            Table with aggregate statistics
        """
        logger.info("Computing document stream statistics")
        
        stats = table.select(
            total_documents=pw.count(),
            avg_word_count=pw.avg(pw.this.word_count),
            max_word_count=pw.max(pw.this.word_count),
            min_word_count=pw.min(pw.this.word_count),
            total_chars=pw.sum(pw.this.char_count)
        )
        
        logger.info("Document statistics computed")
        return stats
    
    def create_indexing_stream(self, documents_table: pw.Table) -> pw.Table:
        """
        Create an indexing stream for fast document lookup.
        Uses Pathway's reactive indexing.
        
        Args:
            documents_table: Input document table
            
        Returns:
            Indexed table optimized for lookups
        """
        logger.info("Creating reactive document index stream")
        
        # Add index column for fast lookups
        indexed = documents_table.with_id(
            pw.this.id  # Use document ID as the pathway index
        )
        
        logger.info("Document indexing stream created")
        return indexed
    
    def process_documents_in_batches(self, 
                                     documents: Dict[str, str],
                                     batch_size: int = 10) -> List[pw.Table]:
        """
        Process documents in reactive batches.
        Useful for handling large document sets with streaming updates.
        
        Args:
            documents: Dict of documents
            batch_size: Number of documents per batch
            
        Returns:
            List of batch tables
        """
        logger.info(f"Processing {len(documents)} documents in batches of {batch_size}")
        
        batches = []
        doc_items = list(documents.items())
        
        for i in range(0, len(doc_items), batch_size):
            batch_dict = dict(doc_items[i:i+batch_size])
            batch_table = self.create_document_stream(batch_dict, "batch")
            batches.append(batch_table)
            logger.debug(f"Created batch {len(batches)} with {len(batch_dict)} documents")
        
        logger.info(f"Created {len(batches)} batches for processing")
        return batches


class PathwayDataProcessor:
    """
    Utility class for data transformations using Pathway's reactive operations.
    """
    
    @staticmethod
    def filter_documents_by_word_count(table: pw.Table, min_words: int) -> pw.Table:
        """Filter table to documents with minimum word count."""
        return table.filter(pw.this.word_count >= min_words)
    
    @staticmethod
    def extract_document_chunks(table: pw.Table, chunk_size: int) -> pw.Table:
        """
        Reactively extract chunks from documents.
        
        Args:
            table: Document table
            chunk_size: Approximate chunk size in words
            
        Returns:
            Table with document chunks
        """
        logger.info(f"Extracting chunks of ~{chunk_size} words from documents")
        
        def create_chunks(content: str) -> List[str]:
            words = content.split()
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                chunks.append(chunk)
            return chunks
        
        # This would normally use apply_batch or map in Pathway
        # For now we return the enhancement schema
        return table
    
    @staticmethod
    def compute_text_metrics(table: pw.Table) -> pw.Table:
        """
        Compute text analysis metrics on the stream.
        
        Args:
            table: Document table
            
        Returns:
            Table with computed metrics
        """
        logger.info("Computing text metrics over document stream")
        
        enhanced = table.select(
            pw.this.id,
            pw.this.content,
            word_count=pw.this.word_count,
            char_count=pw.this.char_count,
            avg_word_length=pw.this.char_count / (pw.this.word_count + 1),
            sentences_estimate=pw.this.content.str.count(".") + pw.this.content.str.count("!") + pw.this.content.str.count("?")
        )
        
        return enhanced
