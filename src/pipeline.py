"""
pipeline.py
Pipeline orchestrating document loading, chunking, claim extraction,
evidence retrieval, and verification for backstory consistency checking.
"""

import logging
from typing import Tuple, Dict, List
from pathlib import Path

from .ingest import NarrativeLoader, BackstoryLoader, PathwayDocumentStore
from .chunk import NarrativeChunker, ChunkIndex
from .claims import ClaimExtractor, ClaimValidator
from .retrieve import EvidenceRetriever
from .judge import ConsistencyJudge, DecisionAggregator
from config import Config

logger = logging.getLogger(__name__)


class ConsistencyCheckPipeline:
    """Pipeline for consistency checking."""

    def __init__(self, narratives_dir: Path, backstories_dir: Path, api_key: str | None = None):
        """Initialize pipeline components."""
        self.narratives_dir = Path(narratives_dir)
        self.backstories_dir = Path(backstories_dir)
        self.api_key = api_key or Config.GEMINI_API_KEY

        self.narrative_loader = NarrativeLoader(narratives_dir)
        self.backstory_loader = BackstoryLoader(backstories_dir)
        self.chunker = NarrativeChunker(chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP)
        self.claim_extractor = ClaimExtractor(api_key=self.api_key)
        self.judge = ConsistencyJudge()
        self.decision_aggregator = DecisionAggregator()

        logger.info("ConsistencyCheckPipeline initialized")

    def process_story(self, story_id: str, verbose: bool = False) -> Tuple[int, str, Dict]:
        """Process story pipeline stages."""
        logger.info("=" * 60)
        logger.info(f"Processing story: {story_id}")
        logger.info("=" * 60)

        metadata = {"story_id": story_id}

        try:
            # Stage 1: Load documents
            logger.info("[1/6] Loading documents...")
            narrative = self.narrative_loader.load_narrative(story_id)
            backstory = self.backstory_loader.load_backstory(story_id)

            narrative_meta = self.narrative_loader.get_narrative_metadata(narrative)
            metadata.update(narrative_meta)

            logger.info(f"Narrative: {narrative_meta['word_count']:,} words")
            logger.info(f"Backstory: {len(backstory.split())} words")

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

            contradictions = sum(1 for r in results if r.get("verdict", "").upper() == "CONTRADICTION")
            metadata["contradictions_found"] = contradictions

            # Stage 6: Aggregate decision
            logger.info("[6/6] Aggregating decision...")
            prediction, rationale = self.decision_aggregator.make_decision(results)
            metadata["verification_results"] = len(results)

            logger.info(f"\nFINAL DECISION: {'INCONSISTENT' if prediction == 0 else 'CONSISTENT'}")
            logger.info(f"RATIONALE: {rationale}")
            logger.info("=" * 60 + "\n")

            return prediction, rationale, metadata

        except Exception as e:
            logger.error(f"Pipeline failed for story {story_id}: {e}", exc_info=True)
            error_rationale = f"Processing error: {str(e)[:100]}"
            return 1, error_rationale, metadata

    def process_batch(self, story_ids: List[str], verbose: bool = False) -> List[Dict]:
        """Process multiple stories."""
        results = []

        for i, story_id in enumerate(story_ids, 1):
            logger.info(f"\n{'#' * 60}")
            logger.info(f"Processing story {i}/{len(story_ids)}: {story_id}")
            logger.info(f"{'#' * 60}\n")

            prediction, rationale, metadata = self.process_story(story_id, verbose)

            results.append({
                "story_id": story_id,
                "prediction": prediction,
                "rationale": rationale,
                "metadata": metadata
            })

        return results


class PathwayIntegrationPipeline(ConsistencyCheckPipeline):
    """Pipeline with Pathway integration."""

    def __init__(self, narratives_dir: Path, backstories_dir: Path, api_key: str | None = None):
        """Initialize with Pathway document store."""
        super().__init__(narratives_dir, backstories_dir, api_key)
        self.document_store = PathwayDocumentStore()
        logger.info("PathwayIntegrationPipeline initialized with document store")


class PipelineFactory:
    """Factory for creating pipelines."""

    @staticmethod
    def create_standard_pipeline(config: Config | None = None) -> ConsistencyCheckPipeline:
        """Create standard pipeline."""
        if config is None:
            config = Config

        return ConsistencyCheckPipeline(
            narratives_dir=config.NARRATIVES_DIR,
            backstories_dir=config.BACKSTORIES_DIR,
            api_key=config.GEMINI_API_KEY
        )

    @staticmethod
    def create_pathway_pipeline(config: Config | None = None) -> PathwayIntegrationPipeline:
        """Create Pathway-integrated pipeline."""
        if config is None:
            config = Config

        return PathwayIntegrationPipeline(
            narratives_dir=config.NARRATIVES_DIR,
            backstories_dir=config.BACKSTORIES_DIR,
            api_key=config.GEMINI_API_KEY
        )


class PipelineValidator:
    """Validate configuration and dependencies."""

    @staticmethod
    def validate_environment() -> Tuple[bool, List[str]]:
        """Check environment configuration."""
        issues = []

        if not Config.GEMINI_API_KEY:
            issues.append("GEMINI_API_KEY not set in environment")

        if not Config.NARRATIVES_DIR.exists():
            issues.append(f"Narratives directory not found: {Config.NARRATIVES_DIR}")

        if not Config.BACKSTORIES_DIR.exists():
            issues.append(f"Backstories directory not found: {Config.BACKSTORIES_DIR}")

        return len(issues) == 0, issues

    @staticmethod
    def print_environment_status() -> bool:
        """Print validation status."""
        is_valid, issues = PipelineValidator.validate_environment()

        if is_valid:
            logger.info("✓ Environment validation passed")
            logger.info(f"  Narratives dir: {Config.NARRATIVES_DIR}")
            logger.info(f"  Backstories dir: {Config.BACKSTORIES_DIR}")
            logger.info(f"  API key: Set" if Config.GEMINI_API_KEY else "  API key: Not set")
        else:
            logger.error("✗ Environment validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")

        return is_valid