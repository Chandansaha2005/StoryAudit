"""
pipeline.py
End-to-end pipeline orchestrating all components
"""

import logging
from typing import Tuple, Dict, List
from pathlib import Path

from .ingest import NarrativeLoader, BackstoryLoader, PathwayDocumentStore
from .chunk import NarrativeChunker, ChunkIndex
from .claims import ClaimExtractor, ClaimValidator
from .retrieve import EvidenceRetriever, EvidenceAggregator, PathwayEvidenceIndex
from .judge import ConsistencyJudge, DecisionAggregator, VerificationResult
from config import Config

logger = logging.getLogger(__name__)


class ConsistencyCheckPipeline:
    """
    Complete pipeline for backstory consistency checking.
    
    Pipeline stages:
    1. Load narrative and backstory (Pathway-based ingestion)
    2. Chunk narrative with temporal ordering
    3. Extract claims from backstory
    4. Retrieve evidence for each claim
    5. Verify each claim against evidence
    6. Aggregate results into final decision
    """
    
    def __init__(self, narratives_dir: Path, backstories_dir: Path, api_key: str = None):
        """
        Initialize pipeline with data directories.
        
        Args:
            narratives_dir: Directory containing narrative files
            backstories_dir: Directory containing backstory files
            api_key: Anthropic API key
        """
        self.narratives_dir = narratives_dir
        self.backstories_dir = backstories_dir
        self.api_key = api_key or Config.ANTHROPIC_API_KEY
        
        # Initialize loaders
        self.narrative_loader = NarrativeLoader(narratives_dir)
        self.backstory_loader = BackstoryLoader(backstories_dir)
        
        # Initialize processing components
        self.chunker = NarrativeChunker(
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP
        )
        self.claim_extractor = ClaimExtractor(api_key=self.api_key)
        self.judge = ConsistencyJudge(api_key=self.api_key)
        self.decision_aggregator = DecisionAggregator()
        
        logger.info("ConsistencyCheckPipeline initialized")
    
    def process_story(self, story_id: str, verbose: bool = False) -> Tuple[int, str, Dict]:
        """
        Process a single story through the complete pipeline.
        
        Args:
            story_id: Story identifier
            verbose: Whether to log detailed progress
            
        Returns:
            Tuple of (prediction, rationale, metadata)
            prediction: 1 = consistent, 0 = inconsistent
            rationale: Brief explanation
            metadata: Additional information about processing
        """
        logger.info(f"{'='*60}")
        logger.info(f"Processing story: {story_id}")
        logger.info(f"{'='*60}")
        
        metadata = {"story_id": story_id}
        
        try:
            # Stage 1: Load documents
            logger.info("[1/6] Loading documents...")
            narrative = self.narrative_loader.load_narrative(story_id)
            backstory = self.backstory_loader.load_backstory(story_id)
            
            narrative_meta = self.narrative_loader.get_narrative_metadata(narrative)
            metadata.update(narrative_meta)
            
            logger.info(f"Loaded narrative: {narrative_meta['word_count']:,} words")
            logger.info(f"Loaded backstory: {len(backstory.split())} words")
            
            # Stage 2: Chunk narrative
            logger.info("[2/6] Chunking narrative...")
            chunks = self.chunker.chunk_narrative(narrative, story_id)
            chunk_index = ChunkIndex(chunks)
            metadata["num_chunks"] = len(chunks)
            
            # Stage 3: Extract claims
            logger.info("[3/6] Extracting backstory claims...")
            claims = self.claim_extractor.extract_claims(backstory, story_id)
            claims = ClaimValidator.validate_claims(claims)
            metadata["num_claims"] = len(claims)
            
            if not claims:
                logger.warning("No valid claims extracted from backstory")
                return 1, "No testable claims found in backstory", metadata
            
            logger.info(f"Extracted {len(claims)} testable claims")
            
            # Stage 4: Retrieve evidence
            logger.info("[4/6] Retrieving evidence for claims...")
            retriever = EvidenceRetriever(chunk_index)
            
            evidence_map = {}
            for claim in claims:
                evidence_chunks = retriever.retrieve_evidence(claim)
                evidence_map[claim.claim_id] = evidence_chunks
            
            # Stage 5: Verify claims
            logger.info("[5/6] Verifying claims against narrative...")
            results = self.judge.verify_claims_batch(claims, evidence_map)
            metadata["verification_results"] = len(results)
            
            # Count contradictions
            contradictions = sum(1 for r in results if r.is_contradiction())
            metadata["contradictions_found"] = contradictions
            
            # Stage 6: Make final decision
            logger.info("[6/6] Aggregating results and making decision...")
            prediction, rationale = self.decision_aggregator.make_decision(results)
            
            # Log detailed report if verbose
            if verbose:
                report = self.decision_aggregator.generate_detailed_report(results)
                logger.info(f"\n{report}")
            
            logger.info(f"\nFINAL DECISION: {prediction}")
            logger.info(f"RATIONALE: {rationale}")
            logger.info(f"{'='*60}\n")
            
            return prediction, rationale, metadata
            
        except Exception as e:
            logger.error(f"Pipeline failed for story {story_id}: {e}", exc_info=True)
            
            # On error, default to consistent with explanation
            error_rationale = f"Processing error: {str(e)[:100]}"
            return 1, error_rationale, metadata
    
    def process_batch(self, story_ids: List[str], 
                     verbose: bool = False) -> List[Dict]:
        """
        Process multiple stories in batch.
        
        Args:
            story_ids: List of story identifiers
            verbose: Whether to log detailed progress
            
        Returns:
            List of result dicts with keys: story_id, prediction, rationale
        """
        results = []
        
        for i, story_id in enumerate(story_ids, 1):
            logger.info(f"\n{'#'*60}")
            logger.info(f"Processing story {i}/{len(story_ids)}: {story_id}")
            logger.info(f"{'#'*60}\n")
            
            prediction, rationale, metadata = self.process_story(story_id, verbose)
            
            results.append({
                "story_id": story_id,
                "prediction": prediction,
                "rationale": rationale,
                "metadata": metadata
            })
        
        return results


class PathwayIntegrationPipeline(ConsistencyCheckPipeline):
    """
    Enhanced pipeline with deeper Pathway integration.
    Demonstrates Pathway's document streaming and indexing capabilities.
    """
    
    def __init__(self, narratives_dir: Path, backstories_dir: Path, api_key: str = None):
        """Initialize pipeline with Pathway components."""
        super().__init__(narratives_dir, backstories_dir, api_key)
        
        # Add Pathway document store
        self.document_store = PathwayDocumentStore()
        
        logger.info("PathwayIntegrationPipeline initialized with document store")
    
    def process_story_with_pathway(self, story_id: str, 
                                   verbose: bool = False) -> Tuple[int, str, Dict]:
        """
        Process story using Pathway document store for management.
        
        Args:
            story_id: Story identifier
            verbose: Whether to log detailed progress
            
        Returns:
            Tuple of (prediction, rationale, metadata)
        """
        logger.info(f"Processing with Pathway integration: {story_id}")
        
        # Load documents into Pathway store
        narrative = self.narrative_loader.load_narrative(story_id)
        backstory = self.backstory_loader.load_backstory(story_id)
        
        # Add to Pathway document store
        self.document_store.add_document(
            f"{story_id}_narrative",
            narrative,
            {"type": "narrative", "story_id": story_id}
        )
        
        self.document_store.add_document(
            f"{story_id}_backstory",
            backstory,
            {"type": "backstory", "story_id": story_id}
        )
        
        # Build Pathway table
        self.document_store.build_pathway_table()
        
        # Continue with standard processing
        return self.process_story(story_id, verbose)


class PipelineFactory:
    """
    Factory for creating configured pipelines.
    """
    
    @staticmethod
    def create_standard_pipeline(config: Config = None) -> ConsistencyCheckPipeline:
        """
        Create standard pipeline with default configuration.
        
        Args:
            config: Configuration object (defaults to Config)
            
        Returns:
            Configured ConsistencyCheckPipeline
        """
        if config is None:
            config = Config
        
        return ConsistencyCheckPipeline(
            narratives_dir=config.NARRATIVES_DIR,
            backstories_dir=config.BACKSTORIES_DIR,
            api_key=config.GOOGLE_API_KEY
        )
    
    @staticmethod
    def create_pathway_pipeline(config: Config = None) -> PathwayIntegrationPipeline:
        """
        Create pipeline with enhanced Pathway integration.
        
        Args:
            config: Configuration object (defaults to Config)
            
        Returns:
            Configured PathwayIntegrationPipeline
        """
        if config is None:
            config = Config
        
        return PathwayIntegrationPipeline(
            narratives_dir=config.NARRATIVES_DIR,
            backstories_dir=config.BACKSTORIES_DIR,
            api_key=config.GOOGLE_API_KEY
        )


class PipelineValidator:
    """
    Validates pipeline configuration and dependencies.
    """
    
    @staticmethod
    def validate_environment() -> Tuple[bool, List[str]]:
        """
        Validate that environment is properly configured.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check API key
        if not Config.GOOGLE_API_KEY:
            issues.append("GOOGLE_API_KEY not set in environment")
        
        # Check data directories
        if not Config.NARRATIVES_DIR.exists():
            issues.append(f"Narratives directory not found: {Config.NARRATIVES_DIR}")
        
        if not Config.BACKSTORIES_DIR.exists():
            issues.append(f"Backstories directory not found: {Config.BACKSTORIES_DIR}")
        
        # Check for sample data
        narrative_files = list(Config.NARRATIVES_DIR.glob("*.txt"))
        backstory_files = list(Config.BACKSTORIES_DIR.glob("*.txt"))
        
        if not narrative_files:
            issues.append("No narrative files found in narratives directory")
        
        if not backstory_files:
            issues.append("No backstory files found in backstories directory")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    @staticmethod
    def print_environment_status():
        """Print environment validation status."""
        is_valid, issues = PipelineValidator.validate_environment()
        
        if is_valid:
            logger.info("✓ Environment validation passed")
            logger.info(f"  Narratives dir: {Config.NARRATIVES_DIR}")
            logger.info(f"  Backstories dir: {Config.BACKSTORIES_DIR}")
            logger.info(f"  API key: {'Set' if Config.GOOGLE_API_KEY else 'Not set'}")
        else:
            logger.error("✗ Environment validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
        
        return is_valid