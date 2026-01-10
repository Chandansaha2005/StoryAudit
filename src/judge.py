"""
judge.py
Minimal consistency judge and aggregator stubs for pipeline integration.
This file provides small, deterministic defaults so the pipeline can run
when LLMs or full verification logic are not available in the environment.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    claim_id: str
    verdict: str
    confidence: float
    reasoning: str


class ConsistencyJudge:
    """Judge returning default verification results."""

    def __init__(self, model: str = "local"):
        self.model = model
        logger.info(f"ConsistencyJudge initialized (model={model})")

    def verify_claim(self, claim, evidence: str) -> dict:
        """Return verification dict."""
        return {
            "claim": getattr(claim, "text", str(claim)),
            "verdict": "CONSISTENT",
            "confidence": 0.4,
            "reasoning": "Default verification (no LLM run)",
            "key_evidence": ""
        }

    def verify_claims_batch(self, claims: List, evidence_map: Dict[str, List]) -> List[dict]:
        return [self.verify_claim(c, "") for c in claims]


class DecisionAggregator:
    """Aggregate results into binary decision."""

    def __init__(self, contradiction_threshold: float = 0.8):
        self.threshold = contradiction_threshold

    def make_decision(self, results: List[dict]) -> Tuple[int, str]:
        # Simple rule: any result with verdict CONTRADICTION and confidence >= threshold => inconsistent
        for r in results:
            if r.get("verdict", "").upper() == "CONTRADICTION" and float(r.get("confidence", 0)) >= self.threshold:
                return 0, "Contradiction detected"
        return 1, "Backstory consistent"
