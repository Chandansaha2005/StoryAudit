"""
csv_processor.py
Process backstory consistency from CSV files with associated novel texts.
Handles test/train CSV formats and maps to narrative files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import re

logger = logging.getLogger(__name__)


class CSVBackstoryDataset:
    """Load and process backstory consistency data from CSV files."""
    
    def __init__(self, csv_path: Path, narratives_dir: Path):
        """
        Initialize dataset from CSV.
        
        Args:
            csv_path: Path to test.csv or train.csv
            narratives_dir: Directory containing novel .txt files
        """
        self.csv_path = csv_path
        self.narratives_dir = narratives_dir
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Load narratives into memory (keyed by book_name)
        self.narratives_cache = self._load_narratives()
        
        logger.info(f"Loaded {len(self.df)} backstory examples from {csv_path.name}")
        logger.info(f"Loaded {len(self.narratives_cache)} narrative texts")
    
    def _load_narratives(self) -> Dict[str, str]:
        """
        Load all novel texts from narratives directory.
        
        Returns:
            Dict mapping book_name -> full text
        """
        narratives = {}
        
        for txt_file in self.narratives_dir.glob("*.txt"):
            book_name = txt_file.stem  # filename without .txt
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                narratives[book_name] = content
                logger.debug(f"Loaded {book_name}: {len(content)} characters")
            except Exception as e:
                logger.error(f"Failed to load {txt_file}: {e}")
        
        return narratives
    
    def get_example(self, idx: int) -> Dict:
        """
        Get a single backstory example with its narrative.
        
        Args:
            idx: Row index in CSV
            
        Returns:
            Dict with:
                - id: Example ID
                - book_name: Novel name
                - character: Character name
                - backstory: Backstory text
                - narrative: Full novel text
                - label: (optional) 'consistent' or 'contradict'
        """
        row = self.df.iloc[idx]
        
        book_name = row['book_name']
        
        # Get narrative text (may not be in cache if filename doesn't match)
        narrative_text = self.narratives_cache.get(book_name, None)
        
        # Try fuzzy matching if exact match fails
        if narrative_text is None:
            narrative_text = self._fuzzy_match_narrative(book_name)
        
        example = {
            'id': row['id'],
            'book_name': book_name,
            'character': row.get('char', ''),
            'caption': row.get('caption', ''),
            'backstory': row['content'],
            'narrative': narrative_text,
            'has_label': 'label' in row and pd.notna(row['label'])
        }
        
        if example['has_label']:
            example['label'] = row['label']
        
        return example
    
    def _fuzzy_match_narrative(self, book_name: str) -> Optional[str]:
        """
        Try fuzzy matching to find narrative if exact match fails.
        
        Args:
            book_name: Book name from CSV
            
        Returns:
            Narrative text or None
        """
        # Try exact case-insensitive match
        for loaded_name, text in self.narratives_cache.items():
            if loaded_name.lower() == book_name.lower():
                return text
        
        # Try partial match (substring)
        for loaded_name, text in self.narratives_cache.items():
            if book_name.lower() in loaded_name.lower() or \
               loaded_name.lower() in book_name.lower():
                logger.debug(f"Fuzzy matched '{book_name}' to '{loaded_name}'")
                return text
        
        logger.warning(f"Could not find narrative for book: {book_name}")
        return None
    
    def get_all_examples(self) -> List[Dict]:
        """Get all backstory examples."""
        examples = []
        for i in range(len(self.df)):
            examples.append(self.get_example(i))
        return examples
    
    def get_labeled_examples(self) -> List[Dict]:
        """Get only examples with labels (for training)."""
        examples = []
        for i in range(len(self.df)):
            example = self.get_example(i)
            if example['has_label']:
                examples.append(example)
        return examples
    
    def get_unlabeled_examples(self) -> List[Dict]:
        """Get only examples without labels (for testing)."""
        examples = []
        for i in range(len(self.df)):
            example = self.get_example(i)
            if not example['has_label']:
                examples.append(example)
        return examples


class ConsistencyCSVProcessor:
    """
    Process backstory consistency for CSV-based examples.
    
    Supports both standard and optimized batch processing with caching.
    """
    
    def __init__(self, pipeline):
        """
        Initialize processor with a consistency pipeline.
        
        Args:
            pipeline: ConsistencyCheckPipeline instance
        """
        self.pipeline = pipeline
    
    def process_csv_batch(self, examples: List[Dict], verbose: bool = False, 
                         optimized: bool = False) -> List[Dict]:
        """
        Process batch of backstory examples from CSV.
        
        Args:
            examples: List of example dicts from CSVBackstoryDataset
            verbose: Enable verbose logging
            optimized: Use optimized processing with result caching
            
        Returns:
            List of result dicts with prediction and rationale
        """
        return self._process_standard(examples, verbose, use_cache=optimized)
    
    def _process_standard(self, examples: List[Dict], verbose: bool = False, 
                         use_cache: bool = False) -> List[Dict]:
        """
        Standard sequential processing with optional caching.
        
        When use_cache=True: Checks cache before processing each example,
        dramatically reducing re-processing time.
        """
        results = []
        cache_hits = 0
        cache_misses = 0
        
        for i, example in enumerate(examples):
            if verbose and i > 0 and i % max(1, len(examples) // 10) == 0:
                logger.info(f"Progress: {i+1}/{len(examples)}")
            
            # Check cache if optimization enabled
            if use_cache and hasattr(self.pipeline, 'cache_manager') and self.pipeline.cache_manager:
                example_id = f"{example.get('book_name', '')}_{example.get('id', '')}"
                cached_result = self.pipeline.cache_manager.get_result(example_id)
                if cached_result:
                    results.append(cached_result)
                    cache_hits += 1
                    continue
            
            # Process if not cached
            result = self._process_single_example(example)
            results.append(result)
            
            # Cache result if optimization enabled
            if use_cache and hasattr(self.pipeline, 'cache_manager') and self.pipeline.cache_manager:
                example_id = f"{example.get('book_name', '')}_{example.get('id', '')}"
                self.pipeline.cache_manager.cache_result(example_id, result)
                cache_misses += 1
        
        logger.info(f"Processed {len(results)} examples")
        if use_cache:
            logger.info(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
            if hasattr(self.pipeline, 'cache_manager') and self.pipeline.cache_manager:
                stats = self.pipeline.cache_manager.get_cache_stats()
                logger.info(f"Cache directory: {stats['cache_dir']}")
        
        return results
        
        return results
    
    def _process_single_example(self, example: Dict) -> Dict:
        """
        Process a single backstory example.
        
        Args:
            example: Single example dict
            
        Returns:
            Result dict with id, prediction, rationale
        """
        try:
            example_id = example['id']
            backstory = example['backstory']
            narrative = example['narrative']
            book_name = example['book_name']
            character = example['character']
            
            # If no narrative found, use default prediction
            if narrative is None:
                logger.warning(f"No narrative found for {book_name}, skipping consistency check")
                return {
                    'id': example_id,
                    'book_name': book_name,
                    'character': character,
                    'prediction': -1,  # Unknown
                    'rationale': 'Narrative text not found'
                }
            
            # Use the pipeline to check consistency
            # We need to create a synthetic story_id for the pipeline
            story_key = f"{book_name}_{character}_{example_id}"
            
            # Call pipeline's consistency checking logic
            prediction, rationale, metadata = self.pipeline.check_consistency(
                narrative=narrative,
                backstory=backstory,
                book_name=book_name,
                character=character
            )
            
            return {
                'id': example_id,
                'book_name': book_name,
                'character': character,
                'prediction': prediction,
                'rationale': rationale,
                'metadata': metadata
            }
        
        except Exception as e:
            logger.error(f"Error processing example {example['id']}: {e}", exc_info=True)
            return {
                'id': example.get('id', -1),
                'book_name': example.get('book_name', ''),
                'character': example.get('character', ''),
                'prediction': -1,
                'rationale': f'Error: {str(e)}'
            }


def save_results_csv(results: List[Dict], output_path: Path) -> None:
    """
    Save results to CSV file in the specified format.
    
    Format:
        Story, ID, Prediction, Rationale
        where Prediction: 1 = consistent, 0 = inconsistent
    
    Args:
        results: List of result dicts
        output_path: Path to save CSV
    """
    csv_rows = []
    
    for result in results:
        csv_rows.append({
            'Story': result.get('book_name', ''),
            'ID': result.get('id', ''),
            'Prediction': result.get('prediction', -1),
            'Rationale': result.get('rationale', '')
        })
    
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total examples: {len(results)}")
    
    # Print summary
    valid_predictions = [r['prediction'] for r in results if r['prediction'] in [0, 1]]
    if valid_predictions:
        consistent_count = sum(1 for p in valid_predictions if p == 1)
        inconsistent_count = sum(1 for p in valid_predictions if p == 0)
        logger.info(f"  Consistent (1): {consistent_count}")
        logger.info(f"  Inconsistent (0): {inconsistent_count}")
