__version__ = "1.0.0"
__author__ = "Your Team Name"
__email__ = "your-email@example.com"

# Core components
from .ingest import NarrativeLoader, BackstoryLoader, PathwayDocumentStore
from .chunk import NarrativeChunker, ChunkIndex, Chunk
from .claims import ClaimExtractor, Claim, ClaimValidator
from .retrieve import EvidenceRetriever, PathwayEvidenceIndex
from .judge import ConsistencyJudge, DecisionAggregator, VerificationResult, AdvancedConsistencyJudge, CausalityChecker
from .pipeline import ConsistencyCheckPipeline, PathwayIntegrationPipeline, PipelineFactory
from .pathway_pipeline import PathwayStreamingPipeline, PathwayDataProcessor

# Advanced components (Track A enhancements)
from .embeddings import EmbeddingGenerator, VectorStore, SemanticSimilarityScorer
from .smart_chunk import SmartChunker, ChunkRelationshipTracker
from .scoring import ConsistencyScorer, ScoringResult
from .symbolic_rules import SymbolicValidator
from .evidence_tracker import EvidenceTracker, RetrievalTracker, EvidenceItem, ReasoningStep

__all__ = [
    # Loaders
    'NarrativeLoader',
    'BackstoryLoader',
    'PathwayDocumentStore',
    
    # Chunking
    'NarrativeChunker',
    'ChunkIndex',
    'Chunk',
    'SmartChunker',
    'ChunkRelationshipTracker',
    
    # Claims
    'ClaimExtractor',
    'Claim',
    'ClaimValidator',
    
    # Retrieval
    'EvidenceRetriever',
    'PathwayEvidenceIndex',
    
    # Verification
    'ConsistencyJudge',
    'DecisionAggregator',
    'VerificationResult',
    'AdvancedConsistencyJudge',
    'CausalityChecker',
    
    # Embeddings & Semantic Search
    'EmbeddingGenerator',
    'VectorStore',
    'SemanticSimilarityScorer',
    
    # Scoring
    'ConsistencyScorer',
    'ScoringResult',
    
    # Symbolic Rules
    'SymbolicValidator',
    
    # Evidence Tracking
    'EvidenceTracker',
    'RetrievalTracker',
    'EvidenceItem',
    'ReasoningStep',
    # Pipeline
    'ConsistencyCheckPipeline',
    'PathwayIntegrationPipeline',
    'PipelineFactory',
    
    # Pathway streaming
    'PathwayStreamingPipeline',
    'PathwayDataProcessor',
]