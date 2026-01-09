"""
run.py
Main entry point 
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from config import Config, setup_logging
from src.pipeline import PipelineFactory, PipelineValidator

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="KDSH 2026 Track A: Backstory Consistency Checker"
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
        help='Use enhanced Pathway integration'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate environment and exit'
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
    logger.info("KDSH 2026 Track A: Backstory Consistency Checker")
    logger.info("="*60)
    
    # Validate environment
    if args.validate:
        is_valid = PipelineValidator.print_environment_status()
        sys.exit(0 if is_valid else 1)
    
    # Check environment
    is_valid, issues = PipelineValidator.validate_environment()
    if not is_valid:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("\nRun with --validate flag for detailed diagnostics")
        sys.exit(1)
    
    # Determine which stories to process
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
        logger.error("  --story-id ID       Process single story")
        logger.error("  --story-ids ID1 ID2 Process multiple stories")
        logger.error("  --all              Process all discovered stories")
        sys.exit(1)
    
    if not story_ids:
        logger.error("No stories to process")
        sys.exit(1)
    
    # Create pipeline
    logger.info(f"Initializing pipeline...")
    if args.pathway:
        logger.info("Using enhanced Pathway integration")
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