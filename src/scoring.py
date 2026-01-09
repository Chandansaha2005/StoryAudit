"""
Advanced scoring system for consistency evaluation.
Provides multi-criteria scoring with evidence grading and confidence measures.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of consistency scoring."""
    consistency_score: float  # 0-1: overall consistency
    confidence: float  # 0-1: confidence in score
    evidence_quality: float  # 0-1: quality of supporting evidence
    entity_consistency: float  # 0-1: consistency of named entities
    temporal_consistency: float  # 0-1: consistency of timeline/events
    narrative_coherence: float  # 0-1: coherence of narrative flow
    reasoning_chain: List[str]  # Steps in reasoning
    contradictions: List[Dict]  # Found contradictions
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'consistency_score': round(self.consistency_score, 3),
            'confidence': round(self.confidence, 3),
            'evidence_quality': round(self.evidence_quality, 3),
            'entity_consistency': round(self.entity_consistency, 3),
            'temporal_consistency': round(self.temporal_consistency, 3),
            'narrative_coherence': round(self.narrative_coherence, 3),
            'reasoning_chain': self.reasoning_chain,
            'contradictions': self.contradictions
        }


class ConsistencyScorer:
    """Multi-criteria consistency scoring system."""
    
    def __init__(self):
        """Initialize scorer."""
        logger.info("ConsistencyScorer initialized")
    
    def score_entity_consistency(self, narrative_entities: Dict[str, List[str]], 
                                 backstory_entities: Dict[str, List[str]]) -> float:
        """
        Score consistency of named entities across documents.
        
        Args:
            narrative_entities: Entity types -> list of entity names from narrative
            backstory_entities: Entity types -> list of entity names from backstory
            
        Returns:
            Score 0-1 (1 = perfect consistency)
        """
        if not narrative_entities or not backstory_entities:
            return 0.5  # Neutral if no entities found
        
        total_score = 0.0
        entity_types = set(narrative_entities.keys()) | set(backstory_entities.keys())
        
        for entity_type in entity_types:
            narr_ents = set(narrative_entities.get(entity_type, []))
            back_ents = set(backstory_entities.get(entity_type, []))
            
            if narr_ents or back_ents:
                # Intersection / union (Jaccard similarity)
                intersection = len(narr_ents & back_ents)
                union = len(narr_ents | back_ents)
                type_score = intersection / union if union > 0 else 0.0
                total_score += type_score
            else:
                total_score += 1.0  # Perfect score if neither has entities
        
        avg_score = total_score / len(entity_types) if entity_types else 0.5
        logger.debug(f"Entity consistency score: {avg_score:.3f}")
        return avg_score
    
    def score_temporal_consistency(self, narrative_events: List[str], 
                                   backstory_events: List[str]) -> float:
        """
        Score consistency of temporal ordering and events.
        
        Args:
            narrative_events: Ordered list of events from narrative
            backstory_events: Ordered list of events from backstory
            
        Returns:
            Score 0-1 (1 = temporal consistency)
        """
        if not narrative_events or not backstory_events:
            return 0.5
        
        # Simple check: do events appear in same relative order?
        # Count how many backstory event sequences appear in narrative
        matches = 0
        
        for i in range(len(backstory_events)):
            for j in range(i + 1, min(i + 3, len(backstory_events))):
                # Check if events[i] appears before events[j] in narrative
                try:
                    i_idx = next(k for k, e in enumerate(narrative_events) if backstory_events[i] in e)
                    j_idx = next(k for k, e in enumerate(narrative_events) if backstory_events[j] in e)
                    if i_idx < j_idx:
                        matches += 1
                except StopIteration:
                    pass
        
        max_possible = len(backstory_events) * 2  # Rough estimate
        score = matches / max_possible if max_possible > 0 else 0.5
        
        logger.debug(f"Temporal consistency score: {score:.3f}")
        return max(0.0, min(1.0, score))
    
    def score_evidence_quality(self, claims: List[Dict], evidence: List[Dict]) -> float:
        """
        Score quality of evidence for claims.
        
        Args:
            claims: List of claim dicts with 'claim' text
            evidence: List of evidence dicts with 'text', 'similarity', 'relevance'
            
        Returns:
            Score 0-1 (1 = high quality evidence)
        """
        if not claims or not evidence:
            return 0.0
        
        quality_scores = []
        
        for claim in claims:
            claim_evidence = [e for e in evidence if e.get('claim_id') == claim.get('id')]
            
            if not claim_evidence:
                quality_scores.append(0.0)  # No evidence for claim
            else:
                # Average similarity and relevance of evidence
                similarities = [e.get('similarity', 0.0) for e in claim_evidence]
                relevances = [e.get('relevance', 0.0) for e in claim_evidence]
                
                avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                avg_rel = sum(relevances) / len(relevances) if relevances else 0.0
                
                # Combined score (weighted average)
                quality = (avg_sim * 0.6 + avg_rel * 0.4)
                quality_scores.append(quality)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        logger.debug(f"Evidence quality score: {avg_quality:.3f}")
        return max(0.0, min(1.0, avg_quality))
    
    def score_narrative_coherence(self, narrative_text: str, claims: List[Dict]) -> float:
        """
        Score overall narrative coherence and consistency.
        
        Args:
            narrative_text: Full narrative text
            claims: Extracted claims
            
        Returns:
            Score 0-1
        """
        if not narrative_text or not claims:
            return 0.5
        
        # Simple heuristics:
        # - Longer narrative = more coherent opportunity
        # - More claims = more testable details
        narrative_length = len(narrative_text)
        claims_count = len(claims)
        
        # Normalize: good ratio is ~1 claim per 500 chars
        length_score = min(1.0, narrative_length / 5000)  # Max score at 5000 chars
        density_score = min(1.0, claims_count / 10)  # Max score at 10+ claims
        
        coherence = (length_score * 0.6 + density_score * 0.4)
        logger.debug(f"Narrative coherence score: {coherence:.3f}")
        return max(0.0, min(1.0, coherence))
    
    def compute_composite_score(self, entity_score: float, temporal_score: float,
                               evidence_score: float, coherence_score: float) -> Tuple[float, float]:
        """
        Compute composite consistency score and confidence.
        
        Args:
            entity_score: Entity consistency (0-1)
            temporal_score: Temporal consistency (0-1)
            evidence_score: Evidence quality (0-1)
            coherence_score: Narrative coherence (0-1)
            
        Returns:
            (composite_score, confidence) tuple
        """
        # Weighted average with equal weights
        weights = [0.25, 0.25, 0.25, 0.25]
        scores = [entity_score, temporal_score, evidence_score, coherence_score]
        
        composite = sum(s * w for s, w in zip(scores, weights))
        
        # Confidence based on variance (agreement between metrics)
        mean = composite
        variance = sum(w * (s - mean) ** 2 for s, w in zip(scores, weights))
        
        # High agreement = high confidence
        confidence = max(0.3, 1.0 - variance)  # Min 0.3 confidence
        
        logger.debug(f"Composite score: {composite:.3f}, confidence: {confidence:.3f}")
        return composite, confidence
    
    def score_full_consistency(self, narrative: str, backstory: str,
                              claims: List[Dict], evidence: List[Dict],
                              narrative_entities: Optional[Dict] = None,
                              backstory_entities: Optional[Dict] = None,
                              reasoning_steps: Optional[List[str]] = None) -> ScoringResult:
        """
        Compute full consistency score with all metrics.
        
        Args:
            narrative: Narrative text
            backstory: Backstory text
            claims: Extracted claims
            evidence: Retrieved evidence
            narrative_entities: Optional entities from narrative
            backstory_entities: Optional entities from backstory
            reasoning_steps: Optional reasoning steps to log
            
        Returns:
            ScoringResult with all metrics
        """
        reasoning = reasoning_steps or []
        contradictions = []
        
        # Score individual dimensions
        entity_score = self.score_entity_consistency(
            narrative_entities or {}, 
            backstory_entities or {}
        )
        reasoning.append(f"Entity consistency: {entity_score:.2%}")
        
        # Extract events (simple: look for past tense verbs/action words)
        narrative_events = self._extract_events(narrative)
        backstory_events = self._extract_events(backstory)
        temporal_score = self.score_temporal_consistency(narrative_events, backstory_events)
        reasoning.append(f"Temporal consistency: {temporal_score:.2%}")
        
        evidence_score = self.score_evidence_quality(claims, evidence)
        reasoning.append(f"Evidence quality: {evidence_score:.2%}")
        
        coherence_score = self.score_narrative_coherence(narrative, claims)
        reasoning.append(f"Narrative coherence: {coherence_score:.2%}")
        
        # Compute composite
        consistency, confidence = self.compute_composite_score(
            entity_score, temporal_score, evidence_score, coherence_score
        )
        reasoning.append(f"Overall consistency: {consistency:.2%} (confidence: {confidence:.2%})")
        
        # Find contradictions (if consistency < 0.5)
        if consistency < 0.5:
            contradictions = self._find_contradictions(narrative, backstory, claims)
        
        return ScoringResult(
            consistency_score=consistency,
            confidence=confidence,
            evidence_quality=evidence_score,
            entity_consistency=entity_score,
            temporal_consistency=temporal_score,
            narrative_coherence=coherence_score,
            reasoning_chain=reasoning,
            contradictions=contradictions
        )
    
    def _extract_events(self, text: str) -> List[str]:
        """Simple event extraction from text."""
        import re
        # Look for sentences with past tense indicators
        past_tense_pattern = r'\b(was|were|had|went|came|said|did|made)\b'
        events = []
        
        for sentence in text.split('.'):
            if re.search(past_tense_pattern, sentence, re.IGNORECASE):
                events.append(sentence.strip())
        
        return events[:20]  # Limit to first 20 events
    
    def _find_contradictions(self, narrative: str, backstory: str, claims: List[Dict]) -> List[Dict]:
        """Find potential contradictions between narrative and backstory."""
        contradictions = []
        
        # Simple heuristic: look for negation patterns
        negation_words = ['not', 'no', 'never', 'neither', 'cannot', "can't", "isn't", "weren't"]
        
        for claim in claims[:5]:  # Check first 5 claims
            claim_text = claim.get('claim', '')
            
            # Check if narrative contradicts backstory
            for neg_word in negation_words:
                if neg_word in backstory.lower() and neg_word not in narrative.lower():
                    contradictions.append({
                        'type': 'negation',
                        'claim': claim_text,
                        'issue': f"Backstory contains '{neg_word}' but narrative doesn't",
                        'severity': 'medium'
                    })
                    break
        
        return contradictions
