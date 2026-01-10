"""
run.py
Main entry point 
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd

from config import Config, setup_logging
from src.pipeline import PipelineFactory, PipelineValidator

logger = logging.getLogger(__name__)


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="StoryAudit: Backstory Consistency Checker"
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
    """Extract story IDs from files."""
    story_ids = []
    
    for file_path in narratives_dir.glob("*.txt"):
        # extract numeric id or filename
        name = file_path.stem
        import re
        match = re.search(r'(\d+)', name)
        if match:
            story_id = match.group(1)
            story_ids.append(story_id)
        else:
            story_ids.append(name)
    
    return sorted(set(story_ids))


def save_results(results: List[Dict], output_path: str = None) -> None:
    """Save results to CSV."""
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


def main():
    """Main CLI entry point."""
    setup_logging()
    
    args = parse_args()
    
    logger.info("="*60)
    logger.info("StoryAudit: Backstory Consistency Checker")
    logger.info("="*60)
    
    if args.validate:
        is_valid = PipelineValidator.print_environment_status()
        sys.exit(0 if is_valid else 1)
    
    is_valid, issues = PipelineValidator.validate_environment()
    if not is_valid:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("\nRun with --validate flag for detailed diagnostics")
        sys.exit(1)
    
    # Determine which stories to process
    
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
        logger.info("Using Pathway pipeline")
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