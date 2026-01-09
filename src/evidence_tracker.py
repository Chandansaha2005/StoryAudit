"""
Evidence tracking and chain-of-reasoning system.
Tracks evidence quality, retrieval, and reasoning paths through the pipeline.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """Single piece of evidence for a claim."""
    id: str
    claim_id: str
    text: str
    source: str  # Where evidence came from
    similarity_score: float  # Semantic similarity 0-1
    relevance_score: float  # Human/model assessed relevance 0-1
    quality_grade: str  # 'strong', 'medium', 'weak'
    chunk_index: int = 0  # Which chunk of source document
    position: int = 0  # Character position in source
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReasoningStep:
    """Single step in reasoning chain."""
    step_number: int
    stage: str  # 'extract', 'retrieve', 'verify', 'score'
    description: str
    result: str  # What was concluded
    confidence: float  # 0-1
    details: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EvidenceTracker:
    """Track evidence and reasoning chains for claims."""
    
    def __init__(self):
        """Initialize tracker."""
        self.evidence: Dict[str, List[EvidenceItem]] = {}  # claim_id -> evidence
        self.reasoning_chain: List[ReasoningStep] = []
        self.claim_verdicts: Dict[str, Dict] = {}  # claim_id -> verdict info
        logger.info("EvidenceTracker initialized")
    
    def add_evidence(self, claim_id: str, evidence: EvidenceItem) -> None:
        """
        Add evidence for a claim.
        
        Args:
            claim_id: ID of claim
            evidence: EvidenceItem to add
        """
        if claim_id not in self.evidence:
            self.evidence[claim_id] = []
        
        self.evidence[claim_id].append(evidence)
        logger.debug(f"Added evidence for claim {claim_id}: {evidence.text[:50]}...")
    
    def add_reasoning_step(self, stage: str, description: str, result: str,
                          confidence: float, details: Optional[Dict] = None) -> None:
        """
        Add step to reasoning chain.
        
        Args:
            stage: Pipeline stage ('extract', 'retrieve', 'verify', 'score')
            description: What was done
            result: What was found/concluded
            confidence: Confidence in this step
            details: Optional additional details
        """
        step = ReasoningStep(
            step_number=len(self.reasoning_chain) + 1,
            stage=stage,
            description=description,
            result=result,
            confidence=confidence,
            details=details or {}
        )
        self.reasoning_chain.append(step)
        logger.debug(f"Step {step.step_number}: {description}")
    
    def grade_evidence(self, claim_id: str, evidence_id: str, grade: str) -> None:
        """
        Assign quality grade to evidence.
        
        Args:
            claim_id: ID of claim
            evidence_id: ID of evidence item
            grade: 'strong', 'medium', or 'weak'
        """
        if claim_id in self.evidence:
            for evidence in self.evidence[claim_id]:
                if evidence.id == evidence_id:
                    evidence.quality_grade = grade
                    logger.debug(f"Graded evidence {evidence_id} as {grade}")
                    break
    
    def set_claim_verdict(self, claim_id: str, verdict: str, confidence: float,
                         reasoning: str, supporting_evidence_count: int = 0) -> None:
        """
        Record final verdict for a claim.
        
        Args:
            claim_id: ID of claim
            verdict: 'consistent', 'inconsistent', 'unknown'
            confidence: Confidence in verdict (0-1)
            reasoning: Explanation of verdict
            supporting_evidence_count: Number of supporting evidence pieces
        """
        self.claim_verdicts[claim_id] = {
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning,
            'supporting_evidence': supporting_evidence_count,
            'evidence_count': len(self.evidence.get(claim_id, [])),
            'timestamp': datetime.now().isoformat()
        }
        logger.debug(f"Claim {claim_id} verdict: {verdict} ({confidence:.0%})")
    
    def get_evidence_summary(self, claim_id: str) -> Dict:
        """
        Get summary of evidence for a claim.
        
        Args:
            claim_id: ID of claim
            
        Returns:
            Summary dict with counts and scores
        """
        claim_evidence = self.evidence.get(claim_id, [])
        
        if not claim_evidence:
            return {
                'claim_id': claim_id,
                'evidence_count': 0,
                'avg_similarity': 0.0,
                'avg_relevance': 0.0,
                'strong_count': 0,
                'medium_count': 0,
                'weak_count': 0,
                'best_evidence': None
            }
        
        similarities = [e.similarity_score for e in claim_evidence]
        relevances = [e.relevance_score for e in claim_evidence]
        grades = [e.quality_grade for e in claim_evidence]
        
        return {
            'claim_id': claim_id,
            'evidence_count': len(claim_evidence),
            'avg_similarity': sum(similarities) / len(similarities),
            'avg_relevance': sum(relevances) / len(relevances),
            'strong_count': grades.count('strong'),
            'medium_count': grades.count('medium'),
            'weak_count': grades.count('weak'),
            'best_evidence': {
                'text': claim_evidence[0].text[:100] + '...',
                'similarity': claim_evidence[0].similarity_score,
                'grade': claim_evidence[0].quality_grade
            } if claim_evidence else None
        }
    
    def get_claim_verdict(self, claim_id: str) -> Optional[Dict]:
        """Get verdict for a claim."""
        return self.claim_verdicts.get(claim_id)
    
    def get_reasoning_chain_summary(self) -> List[Dict]:
        """Get summary of reasoning chain."""
        return [step.to_dict() for step in self.reasoning_chain]
    
    def get_full_report(self) -> Dict:
        """
        Get complete tracking report.
        
        Returns:
            Comprehensive report dict
        """
        all_evidence_summaries = [
            self.get_evidence_summary(claim_id)
            for claim_id in self.evidence.keys()
        ]
        
        verdicts_by_type = {
            'consistent': len([v for v in self.claim_verdicts.values() if v['verdict'] == 'consistent']),
            'inconsistent': len([v for v in self.claim_verdicts.values() if v['verdict'] == 'inconsistent']),
            'unknown': len([v for v in self.claim_verdicts.values() if v['verdict'] == 'unknown'])
        }
        
        avg_confidence = (
            sum(v['confidence'] for v in self.claim_verdicts.values()) / len(self.claim_verdicts)
            if self.claim_verdicts else 0.0
        )
        
        return {
            'summary': {
                'total_claims': len(self.evidence),
                'total_verdicts': len(self.claim_verdicts),
                'verdicts_by_type': verdicts_by_type,
                'avg_confidence': round(avg_confidence, 3),
                'reasoning_steps': len(self.reasoning_chain)
            },
            'evidence': {
                'total_pieces': sum(len(e) for e in self.evidence.values()),
                'avg_per_claim': sum(len(e) for e in self.evidence.values()) / max(len(self.evidence), 1),
                'by_claim': all_evidence_summaries
            },
            'reasoning': {
                'steps': [step.to_dict() for step in self.reasoning_chain],
                'stages': list(set(step.stage for step in self.reasoning_chain))
            },
            'verdicts': self.claim_verdicts
        }
    
    def export_json(self, filepath: str) -> None:
        """Export full report to JSON."""
        report = self.get_full_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Exported tracking report to {filepath}")
    
    def clear(self) -> None:
        """Clear all tracked data."""
        self.evidence.clear()
        self.reasoning_chain.clear()
        self.claim_verdicts.clear()
        logger.info("Evidence tracker cleared")


class RetrievalTracker:
    """Track document retrieval operations."""
    
    def __init__(self):
        """Initialize retrieval tracker."""
        self.retrievals: List[Dict] = []
        logger.info("RetrievalTracker initialized")
    
    def log_retrieval(self, query: str, results_count: int,
                     avg_similarity: float, top_chunk_id: Optional[str] = None) -> None:
        """
        Log a retrieval operation.
        
        Args:
            query: Query text
            results_count: Number of results returned
            avg_similarity: Average similarity of results
            top_chunk_id: ID of top result chunk
        """
        self.retrievals.append({
            'query': query[:100],
            'results_count': results_count,
            'avg_similarity': round(avg_similarity, 3),
            'top_chunk': top_chunk_id,
            'timestamp': datetime.now().isoformat()
        })
        logger.debug(f"Logged retrieval: {results_count} results, avg similarity {avg_similarity:.2%}")
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics."""
        if not self.retrievals:
            return {'total_retrievals': 0}
        
        similarities = [r['avg_similarity'] for r in self.retrievals]
        result_counts = [r['results_count'] for r in self.retrievals]
        
        return {
            'total_retrievals': len(self.retrievals),
            'avg_results_per_query': sum(result_counts) / len(result_counts),
            'avg_similarity': sum(similarities) / len(similarities),
            'retrievals': self.retrievals
        }
