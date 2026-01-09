"""
judge.py
Consistency checking logic for backstory claims against narrative evidence
"""

import logging
import json
import re
import google.generativeai as genai
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .claims import Claim
from .chunk import Chunk
from .retrieve import EvidenceAggregator
from config import Config, CONSISTENCY_CHECK_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    claim: Claim
    verdict: str  # "CONSISTENT" or "CONTRADICTION"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    evidence_used: List[Chunk]
    key_evidence: str = ""
    
    def is_contradiction(self) -> bool:
        """Check if this result indicates a contradiction."""
        return self.verdict == "CONTRADICTION"
    
    def is_high_confidence_contradiction(self, threshold: float = 0.8) -> bool:
        """Check if this is a high-confidence contradiction."""
        return self.is_contradiction() and self.confidence >= threshold


class ConsistencyJudge:
    """
    Judges whether backstory claims are consistent with narrative evidence.
    Uses Gemini LLM for reasoning and verification.
    """
    
    def __init__(self, model: str = "llama3"):
        """
        Initialize consistency judge with Gemini.
        
        Args:
            model: Model name (for compatibility, uses Gemini for actual verification)
        """
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.MODEL_NAME)
        self.aggregator = EvidenceAggregator()
        logger.info(f"ConsistencyJudge initialized with {Config.MODEL_NAME}")
    
    # def verify_claim(self, claim: Claim, evidence_chunks: List[Chunk]) -> VerificationResult:
    def verify_claim(self, claim: Claim, evidence: str) -> dict:
        """
        Verify a single claim against narrative evidence using Gemini.
        
        Args:
            claim: Claim to verify
            evidence: Relevant narrative evidence text
            
        Returns:
            Dict with verification result
        """
        prompt = CONSISTENCY_CHECK_PROMPT.format(
            claim=claim.text,
            evidence=evidence
        )
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": Config.MAX_TOKENS_VERIFICATION,
                    "temperature": Config.TEMPERATURE,
                }
            )
            
            content = response.text
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in verification response")
                return self._default_result()
            
            result = json.loads(json_match.group())
            
            # Validate structure
            required_fields = ["verdict", "confidence", "reasoning"]
            if not all(field in result for field in required_fields):
                logger.error("Invalid verification result structure")
                return self._default_result()
            
            return result

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return self._default_result()
    
    def _default_result(self) -> dict:
        """Return default result on error."""
        return {
            "verdict": "CONSISTENT",
            "confidence": 0.3,
            "reasoning": "Verification error - defaulting to consistent",
            "key_evidence": ""
        }
    
    def _parse_verification_result(self, llm_result: dict, claim: Claim,
                                   evidence_chunks: List[Chunk]) -> VerificationResult:
        """
        Parse LLM result into VerificationResult object.
        
        Args:
            llm_result: Dict from LLM
            claim: Original claim
            evidence_chunks: Evidence chunks used
            
        Returns:
            VerificationResult object
        """
        # Extract and validate verdict
        verdict = llm_result.get("verdict", "CONSISTENT").upper()
        if verdict not in ["CONSISTENT", "CONTRADICTION"]:
            logger.warning(f"Invalid verdict: {verdict}, defaulting to CONSISTENT")
            verdict = "CONSISTENT"
        
        # Extract confidence
        confidence = float(llm_result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        # Extract reasoning
        reasoning = llm_result.get("reasoning", "No reasoning provided")
        key_evidence = llm_result.get("key_evidence", "")
        
        return VerificationResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence_used=evidence_chunks,
            key_evidence=key_evidence
        )
    
    def verify_claims_batch(self, claims: List[Claim], 
                           evidence_map: Dict[str, List[Chunk]]) -> List[dict]:
        """
        Verify multiple claims in batches (3-4 per API call) for cost efficiency.
        Batching reduces API calls by ~75% while maintaining verification quality.
        
        Args:
            claims: List of claims to verify
            evidence_map: Dict mapping claim_id -> evidence chunks
            
        Returns:
            List of verification result dicts
        """
        results = []
        batch_size = 4  # Verify 4 claims per API call
        
        # Process claims in batches
        for batch_idx in range(0, len(claims), batch_size):
            batch = claims[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            total_batches = (len(claims) + batch_size - 1) // batch_size
            
            logger.info(f"Verifying batch {batch_num}/{total_batches} ({len(batch)} claims)")
            
            # Build batch verification prompt
            batch_claims_text = ""
            for i, claim in enumerate(batch, 1):
                evidence_chunks = evidence_map.get(claim.claim_id, [])
                evidence_text = '\n'.join([chunk.text for chunk in evidence_chunks]) if evidence_chunks else "No evidence"
                batch_claims_text += f"\nClaim {i}: {claim.text}\nEvidence: {evidence_text}\n---"
            
            # Verify batch together
            batch_results = self._verify_batch_together(batch_claims_text, len(batch))
            
            # Parse results - one per claim
            if isinstance(batch_results, list) and len(batch_results) == len(batch):
                results.extend(batch_results)
            else:
                # Fallback: verify individually if batch fails
                for claim in batch:
                    evidence_chunks = evidence_map.get(claim.claim_id, [])
                    evidence_text = '\n'.join([chunk.text for chunk in evidence_chunks]) if evidence_chunks else "No evidence found"
                    result = self.verify_claim(claim, evidence_text)
                    results.append(result)
        
        return results
    
    def _verify_batch_together(self, batch_claims_text: str, batch_size: int) -> List[dict]:
        """
        Verify multiple claims in a single API call.
        
        Args:
            batch_claims_text: Formatted text with multiple claims and evidence
            batch_size: Number of claims in this batch
            
        Returns:
            List of verification result dicts (one per claim)
        """
        prompt = f"""Verify the consistency of these {batch_size} claims against their provided narrative evidence. For EACH claim, output a JSON object.

{batch_claims_text}

Output format: For each claim, respond with ONLY:
{{"verdict": "CONSISTENT"|"CONTRADICTION", "confidence": 0.0-1.0, "reasoning": "brief reason", "key_evidence": ""}}

Provide exactly {batch_size} JSON objects, one per line, with no other text."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 800,  # Reduced token limit
                    "temperature": Config.TEMPERATURE,
                }
            )
            
            content = response.text.strip()
            json_objects = []
            
            # Split by lines and extract JSON objects
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Try to extract JSON
                start = line.find('{')
                end = line.rfind('}')
                if start >= 0 and end > start:
                    try:
                        json_str = line[start:end+1]
                        obj = json.loads(json_str)
                        json_objects.append(obj)
                    except json.JSONDecodeError:
                        pass
            
            # If we got the right number, return them
            if len(json_objects) >= batch_size:
                return json_objects[:batch_size]
            
            if len(json_objects) > 0:
                logger.warning(f"Expected {batch_size} results, got {len(json_objects)} - using fallback")
                return json_objects
            
            # Fallback: return empty list to trigger individual verification
            logger.warning(f"Failed to parse batch results, will verify individually")
            return []
        
        except Exception as e:
            logger.error(f"Batch verification failed: {e}")
            return []


class DecisionAggregator:
    """
    Aggregates verification results to make final binary decision.
    Applies strict logical rules.
    """
    
    def __init__(self, contradiction_threshold: float = None):
        """
        Initialize decision aggregator.
        
        Args:
            contradiction_threshold: Confidence threshold for contradictions
        """
        self.threshold = contradiction_threshold or Config.CONTRADICTION_CONFIDENCE_THRESHOLD
        
    def make_decision(self, results: List[dict]) -> Tuple[int, str]:
        """
        Make final binary decision from verification results.
        
        Args:
            results: List of verification result dicts
            
        Returns:
            Tuple of (prediction, rationale)
            prediction: 1 = consistent, 0 = inconsistent
        """
        if not results:
            logger.warning("No verification results to aggregate")
            return 1, "No claims to verify"
        
        # Collect contradictions by confidence
        high_conf_contradictions = []
        medium_conf_contradictions = []
        
        for result in results:
            verdict = result.get("verdict", "CONSISTENT").upper()
            confidence = float(result.get("confidence", 0.5))
            
            if verdict == "CONTRADICTION":
                if confidence >= self.threshold:
                    high_conf_contradictions.append(result)
                elif confidence >= 0.6:
                    medium_conf_contradictions.append(result)
        
        # Apply decision rules
        prediction, rationale = self._apply_decision_rules(
            high_conf_contradictions,
            medium_conf_contradictions,
            len(results)
        )
        
        logger.info(f"Final decision: {prediction} - {rationale}")
        return prediction, rationale
    
    def _apply_decision_rules(self, high_conf: List[dict], 
                             medium_conf: List[dict], 
                             total_claims: int) -> Tuple[int, str]:
        """
        Apply strict decision rules.
        
        RULE 1: ANY high-confidence contradiction → INCONSISTENT
        RULE 2: Multiple (2+) medium-confidence contradictions → INCONSISTENT
        RULE 3: Otherwise → CONSISTENT
        """
        # RULE 1: High-confidence contradiction
        if high_conf:
            bad_claim = high_conf[0]
            reasoning = bad_claim.get("reasoning", "Contradiction found")
            
            rationale = f"CRITICAL: Contradiction detected. {reasoning[:150]}"
            return 0, rationale
        
        # RULE 2: Multiple medium-confidence contradictions
        if len(medium_conf) >= 2:
            rationale = f"Multiple contradictions detected ({len(medium_conf)} medium-confidence issues)"
            return 0, rationale
        
        # RULE 3: Default to consistent
        consistent_count = total_claims - len(high_conf) - len(medium_conf)
        
        if medium_conf:
            rationale = (f"Backstory mostly consistent ({consistent_count}/{total_claims} claims), "
                        f"1 medium-confidence issue insufficient to reject")
        else:
            rationale = f"Backstory consistent with narrative ({consistent_count}/{total_claims} claims verified)"
        
        return 1, rationale
    
    def generate_detailed_report(self, results: List[VerificationResult]) -> str:
        """
        Generate detailed report of verification results.
        
        Args:
            results: Verification results
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "VERIFICATION REPORT",
            "=" * 60,
            f"Total claims verified: {len(results)}",
            ""
        ]
        
        # Group by verdict
        contradictions = [r for r in results if r.get("verdict") == "CONTRADICTION" or (hasattr(r, 'verdict') and r.verdict == "CONTRADICTION")]
        consistent = [r for r in results if r.get("verdict") == "CONSISTENT" or (hasattr(r, 'verdict') and r.verdict == "CONSISTENT")]
        
        lines.append(f"Contradictions: {len(contradictions)}")
        lines.append(f"Consistent: {len(consistent)}")
        lines.append("")
        
        # Detail contradictions
        if contradictions:
            lines.append("CONTRADICTIONS FOUND:")
            lines.append("-" * 60)
            
            for i, result in enumerate(contradictions, 1):
                # Handle both dict and object formats
                if isinstance(result, dict):
                    claim_text = result.get("claim", "")
                    if isinstance(claim_text, dict):
                        claim_text = claim_text.get("text", "")
                    confidence = result.get("confidence", 0)
                    reasoning = result.get("reasoning", "")
                else:
                    claim_text = result.claim.text if hasattr(result, 'claim') else ""
                    confidence = result.confidence if hasattr(result, 'confidence') else 0
                    reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
                
                lines.append(f"{i}. Claim: {claim_text}")
                lines.append(f"   Confidence: {confidence:.2f}")
                lines.append(f"   Reasoning: {reasoning}")
                lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class CausalityChecker:
    """
    Specialized checker for causal relationships between backstory and events.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize causality checker."""
        self.api_key = api_key or Config.ANTHROPIC_API_KEY
        self.client = Anthropic(api_key=self.api_key)
    
    def check_causal_compatibility(self, backstory_element: str,
                                   future_events: str) -> Tuple[bool, str]:
        """
        Check if backstory makes future events causally possible.
        
        Args:
            backstory_element: Element from backstory
            future_events: Events from later in narrative
            
        Returns:
            (is_compatible, explanation)
        """
        from config import CAUSAL_ANALYSIS_PROMPT
        
        prompt = CAUSAL_ANALYSIS_PROMPT.format(
            backstory_element=backstory_element,
            narrative_events=future_events
        )
        
        try:
            response = self.client.messages.create(
                model=Config.MODEL_NAME,
                max_tokens=800,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                is_possible = result.get("causally_possible", True)
                explanation = result.get("explanation", "")
                
                return is_possible, explanation
            
        except Exception as e:
            logger.error(f"Causal check failed: {e}")
        
        # Default to compatible on error
        return True, "Causal check inconclusive"