import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config, setup_logging
from pipeline import PipelineFactory, PipelineValidator, AdvancedConsistencyPipeline
from csv_processor import CSVBackstoryDataset, ConsistencyCSVProcessor, save_results_csv

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StoryAudit: Advanced Backstory Consistency Checker (Track A)"
    )
    
    parser.add_argument(
        '--story-ids',
        nargs='+',
        help='List of story IDs to process (e.g., 1 2 3)'
    )
    
    parser.add_argument(
        '--story-id',
        type=str,
        help='Single story ID to process'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all stories in data directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Output CSV file path (default: results.csv)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--pathway',
        action='store_true',
        help='Use enhanced Pathway streaming integration'
    )
    
    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Use advanced Track A pipeline (neural + symbolic reasoning, semantic retrieval)'
    )
    
    parser.add_argument(
        '--evidence',
        action='store_true',
        help='Enable comprehensive evidence tracking and reporting'
    )
    
    parser.add_argument(
        '--symbolic',
        action='store_true',
        help='Use symbolic rule-based validation alongside neural verification'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate environment and exit'
    )
    
    parser.add_argument(
        '--test-csv',
        type=str,
        help='Path to test.csv file for consistency checking'
    )
    
    parser.add_argument(
        '--train-csv',
        type=str,
        help='Path to train.csv file for labeled training data'
    )
    
    parser.add_argument(
        '--optimized',
        action='store_true',
        help='Use optimized batch processing with caching (10-15x faster)'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached data before processing'
    )
    
    return parser.parse_args()


def discover_story_ids(narratives_dir: Path) -> list[str]:
    """
    Discover available story IDs from narratives directory.
    
    Args:
        narratives_dir: Directory containing narrative files
        
    Returns:
        List of story IDs
    """
    story_ids = []
    
    for file_path in narratives_dir.glob("*.txt"):
        # Extract story ID from filename
        # Handles: story_1.txt, 1.txt, narrative_1.txt
        name = file_path.stem  # filename without extension
        
        # Try to extract numeric ID
        import re
        match = re.search(r'(\d+)', name)
        if match:
            story_id = match.group(1)
            story_ids.append(story_id)
        else:
            # Use full filename as ID
            story_ids.append(name)
    
    return sorted(set(story_ids))


def save_results(results: list[dict], output_path: str):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dicts
        output_path: Output file path
    """
    # Extract core fields for CSV
    csv_data = []
    for result in results:
        csv_data.append({
            'Story ID': result['story_id'],
            'Prediction': result['prediction'],
            'Rationale': result['rationale']
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total stories processed: {len(results)}")
    
    # Print summary statistics
    predictions = df['Prediction'].value_counts()
    logger.info(f"Consistent (1): {predictions.get(1, 0)}")
    logger.info(f"Inconsistent (0): {predictions.get(0, 0)}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    logger.info("="*60)
    logger.info("StoryAudit: Advanced Backstory Consistency Checker")
    logger.info("="*60)
    
    # Validate environment
    if args.validate:
        is_valid = PipelineValidator.print_environment_status()
        sys.exit(0 if is_valid else 1)
    
    # Handle CSV-based processing (skip validation since we don't need standard dirs)
    if args.test_csv:
        logger.info("="*60)
        logger.info("CSV-BASED CONSISTENCY CHECKING")
        if args.optimized:
            logger.info("MODE: OPTIMIZED WITH CACHING")
        else:
            logger.info("MODE: STANDARD (sequential)")
        logger.info("="*60)
        
        csv_path = Path(args.test_csv)
        if not csv_path.exists():
            logger.error(f"Test CSV not found: {csv_path}")
            sys.exit(1)
        
        # Clear cache if requested
        if args.clear_cache:
            logger.info("Clearing cache...")
            from cache_manager import CacheManager
            cache_mgr = CacheManager()
            cache_mgr.clear_cache()
            logger.info("Cache cleared")
        
        # Initialize pipeline with caching support
        logger.info("Initializing advanced pipeline for CSV processing...")
        pipeline = AdvancedConsistencyPipeline(
            Config.NARRATIVES_DIR,
            Config.BACKSTORIES_DIR,
            Config.GEMINI_API_KEY,
            enable_caching=True
        )
        
        # Load CSV dataset - use data folder as narratives directory
        narratives_dir = Path(__file__).parent / "data"
        logger.info(f"Loading dataset from {csv_path} with narratives from {narratives_dir}")
        dataset = CSVBackstoryDataset(csv_path, narratives_dir)
        
        # Get examples to process (unlabeled test examples)
        examples = dataset.get_unlabeled_examples()
        if not examples:
            logger.warning("No unlabeled examples in test CSV, using all examples")
            examples = dataset.get_all_examples()
        
        logger.info(f"Processing {len(examples)} examples from CSV...")
        
        # Create processor and run batch
        processor = ConsistencyCSVProcessor(pipeline)
        results = processor.process_csv_batch(
            examples, 
            verbose=args.verbose,
            optimized=args.optimized
        )
        
        # Save results
        output_path = Path(args.output)
        save_results_csv(results, output_path)
        
        logger.info("="*60)
        logger.info("CSV processing complete!")
        logger.info("="*60)
        sys.exit(0)
    
    # Check environment for standard processing
    is_valid, issues = PipelineValidator.validate_environment()
    if not is_valid:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("\nRun with --validate flag for detailed diagnostics")
        sys.exit(1)
    
    # Handle standard story-based processing
    story_ids = []
    
    if args.story_id:
        story_ids = [args.story_id]
    elif args.story_ids:
        story_ids = args.story_ids
    elif args.all:
        story_ids = discover_story_ids(Config.NARRATIVES_DIR)
        logger.info(f"Discovered {len(story_ids)} stories: {story_ids}")
    else:
        logger.error("Please specify stories to process:")
        logger.error("  --test-csv PATH    Process examples from CSV file")
        logger.error("  --story-id ID      Process single story")
        logger.error("  --story-ids ID1 ID2 Process multiple stories")
        logger.error("  --all              Process all discovered stories")
        sys.exit(1)
    
    if not story_ids:
        logger.error("No stories to process")
        sys.exit(1)
    
    # Create pipeline - Advanced mode priority
    logger.info("Initializing pipeline...")
    if args.advanced or args.evidence or args.symbolic:
        logger.info("=" * 60)
        logger.info("STORYAUDIT ADVANCED PIPELINE")
        logger.info("=" * 60)
        logger.info("Features enabled:")
        logger.info("  ✓ Semantic similarity retrieval with embeddings")
        logger.info("  ✓ Multi-criteria consistency scoring")
        logger.info("  ✓ Symbolic rule-based validation")
        logger.info("  ✓ Hybrid neural + symbolic reasoning")
        logger.info("  ✓ Evidence chain tracking")
        logger.info("  ✓ Pathway streaming integration")
        logger.info("=" * 60)
        
        pipeline = AdvancedConsistencyPipeline(
            Config.NARRATIVES_DIR,
            Config.BACKSTORIES_DIR,
            Config.GEMINI_API_KEY
        )
    elif args.pathway:
        logger.info("Using enhanced Pathway streaming integration")
        pipeline = PipelineFactory.create_pathway_pipeline()
    else:
        pipeline = PipelineFactory.create_standard_pipeline()
    
    # Process stories
    logger.info(f"Processing {len(story_ids)} stories...")
    try:
        results = pipeline.process_batch(story_ids, verbose=args.verbose)
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)
    
    # Save results
    save_results(results, args.output)
    
    logger.info("="*60)
    logger.info("Processing complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()