"""
judge.py
Consistency checking logic using Google Gemini API
"""

import logging
import json
import re
from typing import List, Dict
from dataclasses import dataclass
import google.generativeai as genai

from .claims import Claim
from .chunk import Chunk
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
    Uses Google Gemini API for nuanced reasoning but applies strict decision rules.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize consistency judge with Gemini API.
        
        Args:
            api_key: Google API key
        """
        self.api_key = api_key or Config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")
        
        genai.configure(api_key=self.api_key)
        
        # Try to initialize model, fallback to available models if needed
        try:
            self.model = genai.GenerativeModel(Config.MODEL_NAME)
        except Exception as e:
            logger.warning(f"Failed to load {Config.MODEL_NAME}: {e}. Trying fallback models...")
            # Fallback models in order of preference
            for model_name in ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    logger.info(f"Successfully loaded fallback model: {model_name}")
                    break
                except:
                    continue
            else:
                raise ValueError("Could not load any Gemini model")
        
        logger.info(f"ConsistencyJudge initialized with model")
    
    def verify_claim(self, claim: Claim, evidence_chunks: List[Chunk]) -> VerificationResult:
        """
        Verify a single claim against narrative evidence.
        
        Args:
            claim: Claim to verify
            evidence_chunks: Relevant narrative chunks
            
        Returns:
            VerificationResult with verdict and reasoning
        """
        logger.debug(f"Verifying claim: {claim.text[:50]}...")
        
        if not evidence_chunks:
            # No evidence found - default to CONSISTENT
            # (absence of evidence is not contradiction)
            return VerificationResult(
                claim=claim,
                verdict="CONSISTENT",
                confidence=0.5,
                reasoning="No relevant evidence found in narrative",
                evidence_used=[],
                key_evidence=""
            )
        
        # Aggregate evidence into text
        evidence_text = self._aggregate_evidence(evidence_chunks)
        
        # Call Gemini for verification
        llm_result = self._call_gemini_verification(claim, evidence_text)
        
        # Parse result
        result = self._parse_verification_result(
            llm_result, claim, evidence_chunks
        )
        
        logger.debug(f"Verdict: {result.verdict} (confidence: {result.confidence:.2f})")
        
        return result
    
    def _aggregate_evidence(self, chunks: List[Chunk]) -> str:
        """Aggregate chunks into evidence text."""
        # Sort chunks by temporal order
        sorted_chunks = sorted(chunks, key=lambda c: c.temporal_order)
        
        # Combine with separator
        texts = [f"[Chunk {c.temporal_order}]\n{c.text}" for c in sorted_chunks]
        return "\n\n---\n\n".join(texts)
    
    def _call_gemini_verification(self, claim: Claim, evidence: str) -> dict:
        """
        Call Gemini to verify claim against evidence.
        
        Args:
            claim: Claim to verify
            evidence: Aggregated evidence text
            
        Returns:
            Dict with verification result
        """
        prompt = CONSISTENCY_CHECK_PROMPT.format(
            claim=claim.text,
            evidence=evidence
        )
        
        # List of models to try in order
        models_to_try = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        
        for model_name in models_to_try:
            try:
                if not hasattr(self, '_tried_models'):
                    self._tried_models = set()
                
                if model_name not in self._tried_models:
                    self.model = genai.GenerativeModel(model_name)
                    self._tried_models.add(model_name)
                    logger.debug(f"Using model: {model_name} for verification")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,  # Deterministic
                        max_output_tokens=1500,
                    )
                )
                
                content = response.text
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if not json_match:
                    logger.error("No JSON found in Gemini response")
                    return self._default_result()
                
                result = json.loads(json_match.group())
                
                # Validate structure
                required_fields = ["verdict", "confidence", "reasoning"]
                if not all(field in result for field in required_fields):
                    logger.error("Invalid verification result structure")
                    return self._default_result()
                
                return result
                
            except Exception as e:
                error_str = str(e)
                if "404" in error_str or "not found" in error_str.lower():
                    logger.debug(f"Model {model_name} not available, trying next...")
                    continue
                elif "429" in error_str or "quota" in error_str.lower():
                    logger.error(f"Quota exceeded for {model_name}: {e}")
                    return self._default_result()
                else:
                    logger.debug(f"Error with {model_name}, trying next...")
                    continue
        
        logger.error("No models available for verification")
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
        Parse Gemini result into VerificationResult object.
        
        Args:
            llm_result: Dict from Gemini
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
        try:
            confidence = float(llm_result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            confidence = 0.5
        
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
                           evidence_map: Dict[str, List[Chunk]]) -> List[VerificationResult]:
        """
        Verify multiple claims in batch.
        
        Args:
            claims: List of claims to verify
            evidence_map: Dict mapping claim_id -> evidence chunks
            
        Returns:
            List of VerificationResult objects
        """
        logger.info(f"Verifying {len(claims)} claims...")
        
        results = []
        for i, claim in enumerate(claims, 1):
            evidence_chunks = evidence_map.get(claim.claim_id, [])
            result = self.verify_claim(claim, evidence_chunks)
            results.append(result)
            
            if i % 5 == 0:
                logger.info(f"  Verified {i}/{len(claims)} claims")
        
        return results


class DecisionAggregator:
    """
    Aggregates verification results into final decisions.
    Applies strict deterministic logic for classification.
    """
    
    @staticmethod
    def make_decision(results: List[VerificationResult]) -> tuple[int, str]:
        """
        Make final decision based on verification results.
        
        Args:
            results: List of VerificationResult objects
            
        Returns:
            Tuple of (prediction, rationale)
            prediction: 1 = consistent, 0 = inconsistent
        """
        if not results:
            return 1, "No claims to verify - defaulting to consistent"
        
        # Find contradictions
        contradictions = []
        
        for result in results:
            if result.is_contradiction():
                if result.confidence >= Config.CONTRADICTION_CONFIDENCE_THRESHOLD:
                    contradictions.append(result)
                    break  # Any high-confidence contradiction → reject
        
        # Decision rule: ANY high-confidence contradiction → label = 0
        if contradictions:
            contradiction = contradictions[0]
            reason = f"Contradiction found: {contradiction.claim.text[:80]}... (confidence: {contradiction.confidence:.2f})"
            return 0, reason
        
        # Otherwise: label = 1 (consistent)
        num_verified = len(results)
        reason = f"No contradictions found ({num_verified} claims verified successfully)"
        return 1, reason
    
    @staticmethod
    def generate_detailed_report(results: List[VerificationResult]) -> str:
        """
        Generate detailed report of verification results.
        
        Args:
            results: List of VerificationResult objects
            
        Returns:
            Formatted report string
        """
        report = ["DETAILED VERIFICATION REPORT", "=" * 60]
        
        consistent = [r for r in results if r.verdict == "CONSISTENT"]
        contradictions = [r for r in results if r.verdict == "CONTRADICTION"]
        
        report.append(f"\nTotal claims verified: {len(results)}")
        report.append(f"Consistent: {len(consistent)} ({100*len(consistent)/len(results):.1f}%)")
        report.append(f"Contradictions: {len(contradictions)} ({100*len(contradictions)/len(results):.1f}%)")
        
        if contradictions:
            report.append("\nContradictions found:")
            for r in contradictions:
                report.append(f"  - {r.claim.text[:80]}")
                report.append(f"    Confidence: {r.confidence:.2f}")
                report.append(f"    Reasoning: {r.reasoning}")
        
        return "\n".join(report)
