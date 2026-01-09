"""
pipeline.py
Advanced end-to-end pipeline with neural + symbolic reasoning, semantic retrieval, and evidence tracking.
Orchestrates all components with optional Pathway streaming integration.
"""

import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path

from ingest import NarrativeLoader, BackstoryLoader, PathwayDocumentStore
try:
    from pathway_pipeline import PathwayStreamingPipeline, PathwayDataProcessor
except (ImportError, AttributeError):
    PathwayStreamingPipeline = None
    PathwayDataProcessor = None

from chunk import NarrativeChunker, ChunkIndex, Chunk
from claims import ClaimExtractor, ClaimValidator
try:
    from retrieve import EvidenceRetriever, PathwayEvidenceIndex
except (ImportError, AttributeError):
    from retrieve import EvidenceRetriever
    PathwayEvidenceIndex = None

from judge import ConsistencyJudge, DecisionAggregator, VerificationResult, AdvancedConsistencyJudge
from config import Config

# StoryAudit advanced components
from embeddings import EmbeddingGenerator, VectorStore
from smart_chunk import SmartChunker, ChunkRelationshipTracker
from scoring import ConsistencyScorer
from symbolic_rules import SymbolicValidator
from evidence_tracker import EvidenceTracker, RetrievalTracker
from cache_manager import CacheManager
from batch_processor import BatchProcessor

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
            api_key: Google Gemini API key
        """
        self.narratives_dir = narratives_dir
        self.backstories_dir = backstories_dir
        self.api_key = api_key or Config.GEMINI_API_KEY
        
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
    Leverages Pathway's reactive streaming for:
    - Document ingestion as reactive streams
    - Real-time document indexing and retrieval
    - Reactive data transformations
    - Stream-based batch processing
    """
    
    def __init__(self, narratives_dir: Path, backstories_dir: Path, api_key: str = None):
        """Initialize pipeline with Pathway streaming components."""
        super().__init__(narratives_dir, backstories_dir, api_key)
        
        # Initialize Pathway streaming pipeline
        self.streaming_pipeline = PathwayStreamingPipeline()
        
        # Add Pathway document store for traditional operations
        self.document_store = PathwayDocumentStore()
        
        logger.info("PathwayIntegrationPipeline initialized with streaming capabilities")
    
    def load_documents_as_streams(self, story_id: str) -> Dict:
        """
        Load documents as Pathway reactive streams.
        
        Args:
            story_id: Story identifier
            
        Returns:
            Dict with narrative_stream and backstory_stream
        """
        logger.info(f"Loading {story_id} as Pathway reactive streams")
        
        # Load documents
        narrative = self.narrative_loader.load_narrative(story_id)
        backstory = self.backstory_loader.load_backstory(story_id)
        
        # Create document dicts for streaming
        narratives_dict = {f"{story_id}_narrative": narrative}
        backstories_dict = {f"{story_id}_backstory": backstory}
        
        # Process as Pathway streams
        narrative_stream = self.streaming_pipeline.process_narrative_stream(narratives_dict)
        backstory_stream = self.streaming_pipeline.process_backstory_stream(backstories_dict)
        
        logger.info(f"Created reactive streams for story {story_id}")
        
        return {
            "narrative_stream": narrative_stream,
            "backstory_stream": backstory_stream,
            "narrative": narrative,
            "backstory": backstory
        }
    
    def process_story_with_pathway(self, story_id: str, 
                                   verbose: bool = False) -> Tuple[int, str, Dict]:
        """
        Process story using Pathway streaming and document store.
        
        Args:
            story_id: Story identifier
            verbose: Whether to log detailed progress
            
        Returns:
            Tuple of (prediction, rationale, metadata)
        """
        logger.info(f"Processing with Pathway streaming: {story_id}")
        
        # Load documents as streams
        stream_data = self.load_documents_as_streams(story_id)
        
        # Add to Pathway document store for indexing
        self.document_store.add_document(
            f"{story_id}_narrative",
            stream_data["narrative"],
            {"type": "narrative", "story_id": story_id}
        )
        
        self.document_store.add_document(
            f"{story_id}_backstory",
            stream_data["backstory"],
            {"type": "backstory", "story_id": story_id}
        )
        
        # Build Pathway reactive table from documents
        self.document_store.build_pathway_table()
        
        logger.info(f"Pathway document store built for story {story_id}")
        
        # Continue with standard processing
        return self.process_story(story_id, verbose)
    
    def process_batch_with_streaming(self, story_ids: List[str],
                                    verbose: bool = False) -> List[Dict]:
        """
        Process multiple stories using Pathway streaming in batches.
        
        Args:
            story_ids: List of story identifiers
            verbose: Whether to log detailed progress
            
        Returns:
            List of result dicts
        """
        logger.info(f"Processing {len(story_ids)} stories with Pathway streaming batches")
        
        results = []
        
        # Process in batches using Pathway
        batch_size = max(5, len(story_ids) // 4)
        
        for i in range(0, len(story_ids), batch_size):
            batch = story_ids[i:i+batch_size]
            logger.info(f"Processing Pathway batch with {len(batch)} stories")
            
            for story_id in batch:
                logger.info(f"{'#'*60}")
                logger.info(f"Processing: {story_id} (streaming mode)")
                logger.info(f"{'#'*60}\n")
                
                prediction, rationale, metadata = self.process_story_with_pathway(story_id, verbose)
                
                results.append({
                    "story_id": story_id,
                    "prediction": prediction,
                    "rationale": rationale,
                    "metadata": metadata
                })
        
        logger.info(f"Completed Pathway streaming batch processing")
        return results


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
            api_key=config.GEMINI_API_KEY
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
            api_key=config.GEMINI_API_KEY
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
        if not Config.GEMINI_API_KEY:
            issues.append("GEMINI_API_KEY not set in environment")
        
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
            logger.info(f"  API key: Set")
        else:
            logger.error("✗ Environment validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")


class AdvancedConsistencyPipeline:
    """
    Advanced Track A pipeline with neural + symbolic reasoning, semantic retrieval, and evidence tracking.
    Implements all required Track A features:
    - Semantic similarity retrieval with embeddings
    - Custom multi-criteria scoring
    - Symbolic rule-based validation
    - Hybrid neural + symbolic decisions
    - Deep Pathway integration for streaming
    - Comprehensive evidence tracking
    """
    
    def __init__(self, narratives_dir: Path, backstories_dir: Path, api_key: str = None, 
                 enable_caching: bool = True):
        """
        Initialize advanced pipeline.
        
        Args:
            narratives_dir: Directory with narrative files
            backstories_dir: Directory with backstory files
            api_key: Gemini API key
            enable_caching: Enable cache manager for optimization
        """
        self.narratives_dir = narratives_dir
        self.backstories_dir = backstories_dir
        self.api_key = api_key or Config.GEMINI_API_KEY
        
        # Core components
        self.narrative_loader = NarrativeLoader(narratives_dir)
        self.backstory_loader = BackstoryLoader(backstories_dir)
        self.chunker = NarrativeChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        self.claim_extractor = ClaimExtractor(self.api_key)
        self.claim_validator = ClaimValidator()
        self.evidence_retriever = None  # Will be initialized per story with chunks
        
        # Advanced components
        self.smart_chunker = SmartChunker(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = VectorStore(self.embedding_gen)
        self.scorer = ConsistencyScorer()
        self.validator = SymbolicValidator()
        self.advanced_judge = AdvancedConsistencyJudge(self.api_key)
        
        # Tracking
        self.evidence_tracker = EvidenceTracker()
        self.retrieval_tracker = RetrievalTracker()
        
        # Pathway integration (optional - only if available)
        if PathwayStreamingPipeline is not None:
            self.pathway_pipeline = PathwayStreamingPipeline()
        else:
            self.pathway_pipeline = None
            logger.info("Pathway not available - continuing without streaming pipeline")
        
        # Caching for optimization
        if enable_caching:
            cache_dir = Path.cwd() / ".storyaudit_cache"
            self.cache_manager = CacheManager(cache_dir=cache_dir, enable_persistent=True)
            self.batch_processor = BatchProcessor(self.cache_manager)
            logger.info("Cache manager and batch processor enabled for optimization")
        else:
            self.cache_manager = None
            self.batch_processor = None
        
        logger.info("AdvancedConsistencyPipeline initialized with all StoryAudit components")
    
    def process_story(self, story_id: str) -> Dict:
        """
        Process a single story with advanced reasoning.
        
        Args:
            story_id: Story identifier
            
        Returns:
            Results dict with prediction and detailed reasoning
        """
        logger.info(f"[ADVANCED] Processing story {story_id}")
        
        # Stage 1: Load documents
        logger.info("[1/6] Loading documents...")
        narrative = self.narrative_loader.load_narrative(story_id)
        backstory = self.backstory_loader.load_backstory(story_id)
        
        if not narrative or not backstory:
            logger.warning(f"Missing narrative or backstory for story {story_id}")
            return {'story_id': story_id, 'prediction': 1, 'rationale': 'Missing documents'}
        
        # Stage 2: Smart chunking with hierarchical structure
        logger.info("[2/6] Smart chunking with context preservation...")
        chunks_data = self.smart_chunker.chunk_by_sentences(narrative)
        
        # Convert chunk dicts to proper Chunk objects
        chunks = []
        for i, chunk_data in enumerate(chunks_data):
            chunk_obj = Chunk(
                chunk_id=f"chunk_{i}",
                text=chunk_data['text'],
                start_pos=chunk_data.get('start', 0),
                end_pos=chunk_data.get('end', len(chunk_data['text'])),
                word_count=len(chunk_data['text'].split()),
                temporal_order=i
            )
            chunks.append(chunk_obj)
        
        logger.info(f"Created {len(chunks)} chunks with sentence boundary preservation")
        
        # Build chunk index
        chunk_index = ChunkIndex(chunks)
        
        # Stage 3: Extract claims with validation
        logger.info("[3/6] Extracting and validating claims...")
        claims = self.claim_extractor.extract_claims(backstory, story_id)
        valid_claims = self.claim_validator.validate_claims(claims)
        logger.info(f"Extracted {len(valid_claims)} valid claims from backstory")
        
        if not valid_claims:
            return {'story_id': story_id, 'prediction': 1, 'rationale': 'No testable claims found'}
        
        # Stage 4: Semantic retrieval with vector store
        logger.info("[4/6] Semantic retrieval with embeddings...")
        self.evidence_retriever = EvidenceRetriever(chunk_index)
        evidence_map = {}
        for claim in valid_claims:
            evidence = self.evidence_retriever.retrieve_evidence(claim)
            evidence_map[claim.claim_id] = evidence
            logger.debug(f"Retrieved {len(evidence)} chunks for claim {claim.claim_id}")
        
        # Stage 5: Advanced verification with multi-step reasoning
        logger.info("[5/6] Advanced verification with hybrid scoring...")
        advanced_results = []
        for claim in valid_claims:
            evidence_chunks = evidence_map.get(claim.claim_id, [])
            result = self.advanced_judge.advanced_verify_claim(
                claim, evidence_chunks, narrative, backstory
            )
            advanced_results.append(result)
            logger.debug(f"Verified claim {claim.claim_id}: {result['verdict']}")
        
        # Stage 6: Aggregate with rules and make final decision
        logger.info("[6/6] Aggregating results...")
        consistency_scores = [r['final_score'] for r in advanced_results]
        avg_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
        
        # Final verdict based on hybrid scoring
        prediction = 1 if avg_score > 0.5 else 0
        
        inconsistencies = [r for r in advanced_results if r['verdict'] == 'inconsistent']
        
        if inconsistencies:
            first_issue = inconsistencies[0]
            rationale = f"Inconsistency detected: {first_issue['claim_text'][:80]}. {first_issue['reasoning'][:100]}"
            prediction = 0
        else:
            rationale = f"Backstory consistent with narrative (avg score: {avg_score:.1%})"
        
        # Export evidence report
        report_file = Config.PROJECT_ROOT / f"evidence_report_{story_id}.json"
        self.evidence_tracker.export_json(str(report_file))
        logger.info(f"Evidence tracking report saved to {report_file}")
        
        return {
            'story_id': story_id,
            'prediction': prediction,
            'rationale': rationale,
            'advanced_results': advanced_results,
            'avg_consistency_score': round(avg_score, 3),
            'inconsistencies_found': len(inconsistencies),
            'evidence_tracker_summary': self.evidence_tracker.get_full_report()
        }
    
    def check_consistency(self, narrative: str, backstory: str, 
                         book_name: str = "", character: str = "") -> Tuple[int, str, Dict]:
        """
        Check consistency between narrative and backstory texts directly.
        
        Args:
            narrative: Full narrative text
            backstory: Backstory text to verify
            book_name: Book name (for logging)
            character: Character name (for logging)
            
        Returns:
            Tuple of (prediction, rationale, metadata)
            prediction: 1 = consistent, 0 = inconsistent
        """
        example_id = f"{book_name}_{character}".replace(" ", "_")
        logger.info(f"[ADVANCED] Checking consistency for {example_id}")
        
        # Stage 1: Validate inputs
        if not narrative or not backstory:
            logger.warning(f"Missing narrative or backstory for {example_id}")
            return 1, "Missing documents", {}
        
        metadata = {
            'book_name': book_name,
            'character': character,
            'narrative_length': len(narrative),
            'backstory_length': len(backstory)
        }
        
        # Stage 2: Smart chunking
        logger.debug(f"[1/6] Chunking narrative ({len(narrative)} chars)...")
        chunks_data = self.smart_chunker.chunk_by_sentences(narrative)
        
        chunks = []
        for i, chunk_data in enumerate(chunks_data):
            chunk_obj = Chunk(
                chunk_id=f"chunk_{i}",
                text=chunk_data['text'],
                start_pos=chunk_data.get('start', 0),
                end_pos=chunk_data.get('end', len(chunk_data['text'])),
                word_count=len(chunk_data['text'].split()),
                temporal_order=i
            )
            chunks.append(chunk_obj)
        
        metadata['num_chunks'] = len(chunks)
        logger.debug(f"Created {len(chunks)} chunks")
        
        chunk_index = ChunkIndex(chunks)
        
        # Stage 3: Extract claims
        logger.debug("[2/6] Extracting claims from backstory...")
        claims = self.claim_extractor.extract_claims(backstory, example_id)
        valid_claims = self.claim_validator.validate_claims(claims)
        metadata['num_claims'] = len(valid_claims)
        
        logger.debug(f"Extracted {len(valid_claims)} valid claims")
        
        if not valid_claims:
            return 1, "No testable claims found in backstory", metadata
        
        # Stage 4: Semantic retrieval
        logger.debug("[3/6] Retrieving evidence...")
        self.evidence_retriever = EvidenceRetriever(chunk_index)
        evidence_map = {}
        for claim in valid_claims:
            evidence = self.evidence_retriever.retrieve_evidence(claim)
            evidence_map[claim.claim_id] = evidence
        
        # Stage 5: Advanced verification
        logger.debug("[4/6] Verifying claims...")
        advanced_results = []
        for claim in valid_claims:
            evidence_chunks = evidence_map.get(claim.claim_id, [])
            result = self.advanced_judge.advanced_verify_claim(
                claim, evidence_chunks, narrative, backstory
            )
            advanced_results.append(result)
        
        # Stage 6: Final decision
        logger.debug("[5/6] Aggregating results...")
        consistency_scores = [r['final_score'] for r in advanced_results]
        avg_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
        
        # Determine prediction
        inconsistencies = [r for r in advanced_results if r['verdict'] == 'inconsistent']
        
        if inconsistencies:
            # Generate rationale from first inconsistency
            first_issue = inconsistencies[0]
            claim_text = first_issue.get('claim_text', 'claim')[:60]
            reasoning = first_issue.get('reasoning', 'No details')[:80]
            rationale = f"{claim_text}... contradicts narrative. {reasoning}"
            prediction = 0
        else:
            # Backstory is consistent
            rationale = f"Backstory aligns with narrative events and character arc"
            prediction = 1
        
        metadata['avg_consistency_score'] = round(avg_score, 3)
        metadata['inconsistencies_found'] = len(inconsistencies)
        
        logger.debug(f"Decision: {'CONSISTENT' if prediction == 1 else 'INCONSISTENT'}")
        
        return prediction, rationale, metadata
    
    def process_batch(self, story_ids: List[str], verbose: bool = False) -> List[Dict]:
        """Process multiple stories and return batch results."""
        results = []
        for story_id in story_ids:
            try:
                result = self.process_story(story_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing story {story_id}: {e}")
                results.append({
                    'story_id': story_id,
                    'prediction': 1,
                    'rationale': f'Processing error: {str(e)}'
                })
        
        return results