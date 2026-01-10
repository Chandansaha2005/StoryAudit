__version__ = "1.0.0"
__author__ = "Your Team Name"
__email__ = "your-email@example.com"

# core components
from .ingest import NarrativeLoader, BackstoryLoader, PathwayDocumentStore
from .chunk import NarrativeChunker, ChunkIndex, Chunk
from .claims import ClaimExtractor, Claim, ClaimValidator
from .retrieve import EvidenceRetriever, PathwayEvidenceIndex
from .judge import ConsistencyJudge, DecisionAggregator, VerificationResult
from .pipeline import ConsistencyCheckPipeline, PipelineFactory

__all__ = [
    # loaders
    'NarrativeLoader',
    'BackstoryLoader',
    'PathwayDocumentStore',
    
    # chunking
    'NarrativeChunker',
    'ChunkIndex',
    'Chunk',
    
    # claims
    'ClaimExtractor',
    'Claim',
    'ClaimValidator',
    
    # retrieval
    'EvidenceRetriever',
    'PathwayEvidenceIndex',
    
    # verification
    'ConsistencyJudge',
    'DecisionAggregator',
    'VerificationResult',
    
    # pipeline
    'ConsistencyCheckPipeline',
    'PipelineFactory',
]