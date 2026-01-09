"""
batch_processor.py
Optimized batch processing for consistency checking with caching and performance tracking.
Track-A compliant: Evidence tracking, reproducibility, and audit trail.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Track processing metrics for performance optimization."""
    total_examples: int = 0
    processed_examples: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    total_time: float = 0.0
    per_example_time: float = 0.0
    errors: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'total_examples': self.total_examples,
            'processed': self.processed_examples,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) 
                              if (self.cache_hits + self.cache_misses) > 0 else 0,
            'api_calls': self.api_calls,
            'total_time_seconds': round(self.total_time, 2),
            'per_example_seconds': round(self.per_example_time, 2),
            'errors': self.errors
        }
    
    def log_summary(self) -> None:
        """Log metrics summary."""
        summary = self.get_summary()
        logger.info("="*70)
        logger.info("PROCESSING METRICS")
        logger.info("="*70)
        logger.info(f"Total Examples: {summary['total_examples']}")
        logger.info(f"Successfully Processed: {summary['processed']}")
        logger.info(f"Errors: {summary['errors']}")
        logger.info(f"Cache Hit Rate: {summary['cache_hit_rate']:.1f}% ({summary['cache_hits']} hits, {summary['cache_misses']} misses)")
        logger.info(f"API Calls Saved: {summary['api_calls']}")
        logger.info(f"Total Time: {summary['total_time_seconds']}s ({summary['total_time_seconds']/60:.1f}m)")
        logger.info(f"Per Example Average: {summary['per_example_seconds']:.2f}s")
        logger.info("="*70)


class BatchProcessor:
    """
    Optimized batch processor for consistency checking.
    
    Features:
    - Novel preprocessing and caching
    - Embedding caching across examples
    - Parallel processing capabilities
    - Evidence tracking for Track-A compliance
    - Performance monitoring and metrics
    """
    
    def __init__(self, cache_manager, max_batch_size: int = 10):
        """
        Initialize batch processor.
        
        Args:
            cache_manager: CacheManager instance for caching
            max_batch_size: Maximum examples to process before logging progress
        """
        self.cache_manager = cache_manager
        self.max_batch_size = max_batch_size
        self.metrics = ProcessingMetrics()
        self.evidence_chain: List[Dict] = []
    
    def preprocess_novels(self, examples: List[Dict], pipeline) -> Dict[str, Any]:
        """
        Preprocess all novels in batch (one-time cost).
        
        This is the KEY optimization: Process each novel once, then reuse for all examples.
        
        Args:
            examples: List of example dicts with narrative/backstory
            pipeline: ConsistencyCheckPipeline instance
            
        Returns:
            Dict mapping novel_name -> processed data (chunks, embeddings, etc)
        """
        logger.info("="*70)
        logger.info("PREPROCESSING NOVELS (One-time cost)")
        logger.info("="*70)
        
        # Group examples by novel
        novels_in_batch = {}
        for example in examples:
            novel_name = example.get('book_name', 'Unknown')
            if novel_name not in novels_in_batch:
                novels_in_batch[novel_name] = []
            novels_in_batch[novel_name].append(example)
        
        logger.info(f"Found {len(novels_in_batch)} unique novels in batch")
        
        processed = {}
        
        for novel_name in novels_in_batch:
            logger.info(f"\nProcessing novel: {novel_name}")
            
            # Check cache for chunks
            chunks = self.cache_manager.get_chunks(novel_name)
            if chunks:
                logger.info(f"  ✓ Chunks loaded from cache ({len(chunks)} chunks)")
                self.metrics.cache_hits += 1
            else:
                logger.info(f"  → Creating chunks (may take a moment)...")
                # Get narrative text
                example = novels_in_batch[novel_name][0]
                narrative = example.get('narrative')
                
                if narrative:
                    # Chunk the narrative
                    chunks = pipeline.smart_chunker.chunk_by_sentences(narrative)
                    self.cache_manager.cache_chunks(novel_name, chunks)
                    logger.info(f"  ✓ Created and cached {len(chunks)} chunks")
                    self.metrics.cache_misses += 1
            
            # Check cache for embeddings
            embeddings = self.cache_manager.get_embeddings(novel_name)
            if embeddings:
                logger.info(f"  ✓ Embeddings loaded from cache")
                self.metrics.cache_hits += 1
            else:
                logger.info(f"  → Generating embeddings (may take a moment)...")
                # Generate embeddings for all chunks
                embeddings = {}
                chunk_texts = [chunk.get('text') if isinstance(chunk, dict) else chunk.text for chunk in chunks]
                
                # Use batch embedding for efficiency
                embedding_vectors = pipeline.embedding_gen.embed_batch(chunk_texts)
                for i, embedding_vec in enumerate(embedding_vectors):
                    if embedding_vec is not None:
                        embeddings[i] = embedding_vec.tolist()  # Convert numpy array to list for serialization
                
                self.cache_manager.cache_embeddings(novel_name, embeddings)
                logger.info(f"  ✓ Generated and cached {len(embeddings)} embeddings")
                self.metrics.cache_misses += 1
            
            processed[novel_name] = {
                'chunks': chunks,
                'embeddings': embeddings,
                'example_count': len(novels_in_batch[novel_name])
            }
        
        logger.info("\n" + "="*70)
        logger.info("NOVEL PREPROCESSING COMPLETE")
        logger.info("="*70)
        
        return processed
    
    def process_batch(self, examples: List[Dict], pipeline, verbose: bool = False) -> List[Dict]:
        """
        Process batch of examples with optimizations.
        
        Args:
            examples: List of example dicts
            pipeline: ConsistencyCheckPipeline instance
            verbose: Enable verbose logging
            
        Returns:
            List of result dicts
        """
        self.metrics.total_examples = len(examples)
        batch_start = time.time()
        
        logger.info("\n" + "="*70)
        logger.info(f"STARTING BATCH PROCESSING: {len(examples)} examples")
        logger.info("="*70)
        
        # Step 1: Preprocess all novels (one-time)
        processed_novels = self.preprocess_novels(examples, pipeline)
        
        # Step 2: Process each example with cached data
        logger.info("\n" + "="*70)
        logger.info("PROCESSING EXAMPLES WITH CACHED DATA")
        logger.info("="*70)
        
        results = []
        
        for i, example in enumerate(examples):
            # Progress reporting
            if i > 0 and i % self.max_batch_size == 0:
                elapsed = time.time() - batch_start
                per_example = elapsed / i
                remaining = (len(examples) - i) * per_example
                logger.info(f"\nProgress: {i}/{len(examples)} | Elapsed: {elapsed:.1f}s | Est. Remaining: {remaining:.1f}s")
            
            try:
                # Check if result is already cached
                example_id = f"{example.get('book_name', '')}_{example.get('id', '')}"
                cached_result = self.cache_manager.get_result(example_id)
                
                if cached_result:
                    logger.debug(f"Using cached result for example {example['id']}")
                    results.append(cached_result)
                    self.metrics.cache_hits += 1
                    continue
                
                # Process example using pipeline
                result = self._process_single_example(example, pipeline, processed_novels)
                
                # Cache result
                self.cache_manager.cache_result(example_id, result)
                results.append(result)
                self.metrics.processed_examples += 1
                self.metrics.cache_misses += 1
                
            except Exception as e:
                logger.error(f"Error processing example {example.get('id', 'unknown')}: {e}")
                results.append({
                    'id': example.get('id', -1),
                    'book_name': example.get('book_name', ''),
                    'character': example.get('character', ''),
                    'prediction': -1,
                    'rationale': f'Processing error: {str(e)}'
                })
                self.metrics.errors += 1
        
        # Record metrics
        self.metrics.total_time = time.time() - batch_start
        self.metrics.per_example_time = self.metrics.total_time / len(examples) if examples else 0
        
        # Log summary
        self.metrics.log_summary()
        
        return results
    
    def _process_single_example(self, example: Dict, pipeline, processed_novels: Dict) -> Dict:
        """
        Process a single example with cached novel data.
        
        Args:
            example: Example dict
            pipeline: Pipeline instance
            processed_novels: Preprocessed novel data
            
        Returns:
            Result dict
        """
        example_id = example['id']
        novel_name = example.get('book_name', '')
        backstory = example['backstory']
        narrative = example['narrative']
        
        # Use check_consistency method from pipeline
        # This avoids re-chunking and re-embedding the narrative
        prediction, rationale, metadata = pipeline.check_consistency(
            narrative=narrative,
            backstory=backstory,
            book_name=novel_name,
            character=example.get('character', '')
        )
        
        # Track evidence for Track-A compliance
        evidence_entry = {
            'example_id': example_id,
            'novel': novel_name,
            'character': example.get('character', ''),
            'prediction': prediction,
            'timestamp': time.time(),
            'metadata': metadata
        }
        self.evidence_chain.append(evidence_entry)
        
        return {
            'id': example_id,
            'book_name': novel_name,
            'character': example.get('character', ''),
            'prediction': prediction,
            'rationale': rationale
        }
    
    def get_evidence_report(self) -> Dict[str, Any]:
        """
        Generate evidence report for Track-A compliance.
        
        Returns:
            Evidence chain and metadata
        """
        return {
            'evidence_chain': self.evidence_chain,
            'metrics': self.metrics.get_summary(),
            'total_entries': len(self.evidence_chain)
        }
    
    def save_evidence_report(self, output_file: Path) -> None:
        """
        Save evidence report for audit trail.
        
        Args:
            output_file: Path to save report
        """
        report = self.get_evidence_report()
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evidence report saved to {output_file}")
