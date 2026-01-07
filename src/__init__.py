__version__ = "1.0.0"
__author__ = "Your Team Name"
__email__ = "your-email@example.com"

# Core components
from .ingest import NarrativeLoader, BackstoryLoader, PathwayDocumentStore
from .chunk import NarrativeChunker, ChunkIndex, Chunk
from .claims import ClaimExtractor, Claim, ClaimValidator
from .retrieve import EvidenceRetriever, PathwayEvidenceIndex
from .judge import ConsistencyJudge, DecisionAggregator, VerificationResult
from .pipeline import ConsistencyCheckPipeline, PipelineFactory

__all__ = [
    # Loaders
    'NarrativeLoader',
    'BackstoryLoader',
    'PathwayDocumentStore',
    
    # Chunking
    'NarrativeChunker',
    'ChunkIndex',
    'Chunk',
    
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
    
    # Pipeline
    'ConsistencyCheckPipeline',
    'PipelineFactory',
]