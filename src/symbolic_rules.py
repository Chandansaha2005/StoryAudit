"""
Symbolic rules for consistency validation.
Provides rule-based validation of narrative consistency as hybrid with neural scoring.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class SymbolicValidator:
    """Validate consistency using symbolic rules."""
    
    def __init__(self):
        """Initialize validator."""
        logger.info("SymbolicValidator initialized")
        self.violations = []
    
    def validate_entity_consistency(self, narrative_text: str, 
                                    backstory_text: str) -> Tuple[float, List[str]]:
        """
        Validate consistency of named entities using rules.
        
        Args:
            narrative_text: Narrative text
            backstory_text: Backstory text
            
        Returns:
            (score, violations) tuple
        """
        violations = []
        
        # Extract person names from text
        name_pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b'
        narrative_names = set(re.findall(name_pattern, narrative_text))
        backstory_names = set(re.findall(name_pattern, backstory_text))
        
        # Check: backstory shouldn't introduce new primary names
        new_names = backstory_names - narrative_names
        if new_names and len(new_names) > 2:  # Allow up to 2 new supporting characters
            violations.append(f"Backstory introduces {len(new_names)} new primary characters")
        
        # Check: key names should appear in both
        shared_names = narrative_names & backstory_names
        score = len(shared_names) / max(len(narrative_names), 1)
        
        logger.debug(f"Entity consistency: {score:.2%}, violations: {len(violations)}")
        return score, violations
    
    def validate_temporal_consistency(self, narrative_text: str,
                                     backstory_text: str) -> Tuple[float, List[str]]:
        """
        Validate temporal/chronological consistency using rules.
        
        Args:
            narrative_text: Narrative text
            backstory_text: Backstory text
            
        Returns:
            (score, violations) tuple
        """
        violations = []
        
        # Time period keywords
        past_periods = {
            'day before': -1,
            'week before': -7,
            'month before': -30,
            'year before': -365,
            'previously': -999,  # Unspecified past
            'before': -999
        }
        
        future_periods = {
            'next day': 1,
            'soon': 7,
            'later': 999
        }
        
        # Count temporal markers
        narrative_temporal = sum(narrative_text.lower().count(term) for term in past_periods)
        backstory_temporal = sum(backstory_text.lower().count(term) for term in past_periods)
        
        # Check: backstory should have more past references (it's the past)
        if backstory_temporal < narrative_temporal * 0.5:
            violations.append("Backstory has fewer past-tense references than narrative")
        
        # Check: no future references in backstory talking about narrative events
        for future_term in future_periods:
            if future_term in backstory_text.lower():
                # This might be okay if it's predicting future from backstory perspective
                pass
        
        # Scoring: higher temporal consistency = more past refs in backstory
        score = min(1.0, backstory_temporal / max(narrative_temporal, 1))
        
        logger.debug(f"Temporal consistency: {score:.2%}, violations: {len(violations)}")
        return score, violations
    
    def validate_causality_chains(self, narrative_claims: List[str],
                                  backstory_claims: List[str]) -> Tuple[float, List[str]]:
        """
        Validate causal relationships in narrative.
        
        Args:
            narrative_claims: Claims from narrative
            backstory_claims: Claims from backstory
            
        Returns:
            (score, violations) tuple
        """
        violations = []
        
        # Causal words
        causal_markers = ['because', 'therefore', 'as a result', 'caused', 'led to', 
                         'resulted in', 'since', 'due to', 'caused by']
        
        # Count causal relationships
        narrative_causal = sum(1 for claim in narrative_claims 
                             if any(marker in claim.lower() for marker in causal_markers))
        backstory_causal = sum(1 for claim in backstory_claims 
                             if any(marker in claim.lower() for marker in causal_markers))
        
        # Check: backstory should explain causes, narrative shows effects
        if backstory_causal == 0 and len(backstory_claims) > 3:
            violations.append("Backstory lacks causal explanations")
        
        score = min(1.0, backstory_causal / max(narrative_causal, 1)) if narrative_causal > 0 else 0.5
        
        logger.debug(f"Causality consistency: {score:.2%}, violations: {len(violations)}")
        return score, violations
    
    def validate_narrative_coherence_rules(self, text: str) -> Tuple[float, List[str]]:
        """
        Validate narrative coherence using structural rules.
        
        Args:
            text: Text to validate
            
        Returns:
            (score, violations) tuple
        """
        violations = []
        
        # Rule 1: Text should have reasonable length
        if len(text) < 500:
            violations.append("Text is unusually short for a narrative")
        
        # Rule 2: Text should have multiple sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            violations.append(f"Text has only {len(sentences)} sentences (expected >= 5)")
        
        # Rule 3: Variety of sentence lengths (not all too short/long)
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        if avg_length < 3:
            violations.append("Sentences are too short (fragment-like)")
        elif avg_length > 50:
            violations.append("Sentences are too long (run-on style)")
        
        # Rule 4: Use of proper punctuation
        if text.count('.') + text.count('!') + text.count('?') < len(sentences) * 0.7:
            violations.append("Inconsistent punctuation in narrative")
        
        # Scoring based on violations
        score = max(0.0, 1.0 - len(violations) * 0.2)
        
        logger.debug(f"Coherence rules: {score:.2%}, violations: {len(violations)}")
        return score, violations
    
    def validate_pronoun_consistency(self, narrative_text: str,
                                     backstory_text: str) -> Tuple[float, List[str]]:
        """
        Validate consistency of pronoun usage and perspective.
        
        Args:
            narrative_text: Narrative text
            backstory_text: Backstory text
            
        Returns:
            (score, violations) tuple
        """
        violations = []
        
        # Count perspective pronouns
        first_person = ['i ', ' i ', 'me', 'my', 'mine']
        second_person = ['you', 'your', 'yours']
        third_person = ['he', 'she', 'it', 'they', 'his', 'her', 'their']
        
        def count_pronouns(text, pronouns):
            return sum(text.lower().count(p) for p in pronouns)
        
        narrative_1st = count_pronouns(narrative_text, first_person)
        narrative_3rd = count_pronouns(narrative_text, third_person)
        backstory_1st = count_pronouns(backstory_text, first_person)
        backstory_3rd = count_pronouns(backstory_text, third_person)
        
        # Both should use consistent perspective (either both 1st or both 3rd)
        narrative_is_1st = narrative_1st > narrative_3rd
        backstory_is_1st = backstory_1st > backstory_3rd
        
        if narrative_is_1st != backstory_is_1st:
            violations.append("Narrative and backstory use different perspectives (1st vs 3rd person)")
        
        # Score based on consistency
        score = 1.0 if narrative_is_1st == backstory_is_1st else 0.5
        
        logger.debug(f"Pronoun consistency: {score:.2%}, violations: {len(violations)}")
        return score, violations
    
    def validate_negation_logic(self, narrative_text: str,
                               backstory_text: str) -> Tuple[float, List[str]]:
        """
        Check for logical contradictions using negation logic.
        
        Args:
            narrative_text: Narrative text
            backstory_text: Backstory text
            
        Returns:
            (score, violations) tuple
        """
        violations = []
        
        negation_phrases = ["not ", "no ", "never ", "neither ", "cannot ", "couldn't"]
        
        # Extract sentences with negations
        narrative_negations = []
        backstory_negations = []
        
        for sentence in re.split(r'[.!?]', narrative_text):
            if any(neg in sentence.lower() for neg in negation_phrases):
                narrative_negations.append(sentence.strip())
        
        for sentence in re.split(r'[.!?]', backstory_text):
            if any(neg in sentence.lower() for neg in negation_phrases):
                backstory_negations.append(sentence.strip())
        
        # Check for contradictory negations
        # (If narrative says "X never happened", backstory shouldn't say "X happened")
        for narrative_sent in narrative_negations[:5]:  # Sample first 5
            # Simple check: look for key noun from negation
            words = narrative_sent.split()
            for word in words[-5:]:  # Look at last 5 words
                if word.lower() not in negation_phrases and len(word) > 3:
                    # Check if this word appears positively in backstory
                    for backstory_sent in backstory_negations:
                        if word.lower() not in backstory_sent.lower():
                            # Potential contradiction
                            violations.append(f"Potential negation conflict: narrative negates '{word}'")
                            break
        
        # Score: no violations = perfect
        score = max(0.0, 1.0 - len(violations) * 0.3)
        
        logger.debug(f"Negation logic: {score:.2%}, violations: {len(violations)}")
        return score, violations
    
    def hybrid_score(self, neural_score: float, rule_violations: List[str]) -> float:
        """
        Combine neural score with rule-based violations.
        
        Args:
            neural_score: Score from neural model (0-1)
            rule_violations: List of rule violations
            
        Returns:
            Hybrid score (0-1)
        """
        # Penalize for each rule violation
        violation_penalty = len(rule_violations) * 0.1
        adjusted_score = max(0.0, neural_score - violation_penalty)
        
        logger.debug(f"Hybrid score: {adjusted_score:.3f} (neural: {neural_score:.3f}, violations: {len(rule_violations)})")
        return adjusted_score
    
    def validate_all(self, narrative_text: str,
                    backstory_text: str,
                    narrative_claims: List[str],
                    backstory_claims: List[str],
                    neural_score: float = 0.5) -> Dict:
        """
        Run all validations and return comprehensive report.
        
        Args:
            narrative_text: Narrative text
            backstory_text: Backstory text
            narrative_claims: Claims from narrative
            backstory_claims: Claims from backstory
            neural_score: Initial neural consistency score
            
        Returns:
            Validation report dict
        """
        all_violations = []
        
        # Run all validations
        entity_score, entity_viols = self.validate_entity_consistency(narrative_text, backstory_text)
        all_violations.extend([f"[Entity] {v}" for v in entity_viols])
        
        temporal_score, temporal_viols = self.validate_temporal_consistency(narrative_text, backstory_text)
        all_violations.extend([f"[Temporal] {v}" for v in temporal_viols])
        
        causality_score, causality_viols = self.validate_causality_chains(narrative_claims, backstory_claims)
        all_violations.extend([f"[Causality] {v}" for v in causality_viols])
        
        coherence_score, coherence_viols = self.validate_narrative_coherence_rules(narrative_text)
        all_violations.extend([f"[Coherence] {v}" for v in coherence_viols])
        
        pronoun_score, pronoun_viols = self.validate_pronoun_consistency(narrative_text, backstory_text)
        all_violations.extend([f"[Pronoun] {v}" for v in pronoun_viols])
        
        negation_score, negation_viols = self.validate_negation_logic(narrative_text, backstory_text)
        all_violations.extend([f"[Negation] {v}" for v in negation_viols])
        
        # Compute hybrid score
        symbolic_avg = (entity_score + temporal_score + causality_score + coherence_score +
                       pronoun_score + negation_score) / 6.0
        hybrid = self.hybrid_score(neural_score, all_violations)
        
        return {
            'hybrid_score': round(hybrid, 3),
            'symbolic_score': round(symbolic_avg, 3),
            'neural_score': round(neural_score, 3),
            'entity_consistency': round(entity_score, 3),
            'temporal_consistency': round(temporal_score, 3),
            'causality_consistency': round(causality_score, 3),
            'narrative_coherence': round(coherence_score, 3),
            'pronoun_consistency': round(pronoun_score, 3),
            'negation_logic': round(negation_score, 3),
            'violations': all_violations,
            'violation_count': len(all_violations)
        }
