"""
claims.py
Extract atomic, testable claims from character backstories
"""

import logging
import json
import re
from typing import List
from dataclasses import dataclass
import google.generativeai as genai

from config import Config, CLAIM_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """Single testable claim from backstory."""
    claim_id: str
    category: str
    text: str
    importance: str  # high, medium, low
    
    def __repr__(self):
        return f"Claim({self.claim_id}: {self.text[:50]}...)"


class ClaimExtractor:
    """Extract testable claims from text."""
    
    def __init__(self, api_key: str = None):
        """Initialize claim extractor."""
        self.api_key = api_key or Config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Google Gemini API key required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(Config.MODEL_NAME)
        logger.info(f"ClaimExtractor initialized with {Config.MODEL_NAME}")
    
    def extract_claims(self, backstory: str, story_id: str) -> List[Claim]:
        """Extract and prioritize claims."""
        logger.info(f"Extracting claims from backstory ({len(backstory)} chars)")
        
        # call llm for structured claims
        raw_claims = self._call_llm_extraction(backstory)
        
        # parse and validate
        claims = self._parse_claims(raw_claims, story_id)
        
        # prioritize
        claims = self._prioritize_claims(claims)
        
        logger.info(f"Extracted {len(claims)} claims")
        self._log_claim_distribution(claims)
        
        return claims
    
    def _call_llm_extraction(self, backstory: str) -> dict:
        """Call LLM for claim extraction."""
        prompt = CLAIM_EXTRACTION_PROMPT.format(backstory=backstory)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": Config.MAX_TOKENS_EXTRACTION,
                    "temperature": Config.TEMPERATURE,
                }
            )
            
            content = response.text
            
            # extract json from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.error("No JSON found in extraction response")
                return {"claims": []}
            
            result = json.loads(json_match.group())
            
            if "claims" not in result:
                logger.error("Invalid response structure: missing 'claims' key")
                return {"claims": []}
            
            logger.debug(f"LLM extracted {len(result['claims'])} raw claims")
            return result
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            # Return empty but valid structure
            return {"claims": []}
    
    def _parse_claims(self, raw_claims: dict, story_id: str) -> List[Claim]:
        """Parse LLM output into objects."""
        claims = []
        
        for idx, raw_claim in enumerate(raw_claims.get("claims", [])):
            try:
                # validate required fields
                if not isinstance(raw_claim, dict):
                    continue
                
                claim_text = raw_claim.get("claim", "").strip()
                if not claim_text:
                    continue
                
                category = raw_claim.get("category", "unknown")
                importance = raw_claim.get("importance", "medium")
                
                # create claim with generated id
                claim = Claim(
                    claim_id=f"{story_id}_claim_{idx:03d}",
                    category=category,
                    text=claim_text,
                    importance=importance
                )
                
                claims.append(claim)
                
            except Exception as e:
                logger.warning(f"Failed to parse claim {idx}: {e}")
                continue
        
        return claims
    
    def _prioritize_claims(self, claims: List[Claim]) -> List[Claim]:
        """Sort claims by importance score."""
        # define priority scoring
        importance_scores = {"high": 3, "medium": 2, "low": 1}
        
        # categories that create strong constraints
        constraint_categories = {
            "character_events": 3,
            "skills_knowledge": 3,
            "constraints": 3,
            "physical_biological": 2,
            "personality_traits": 2,
            "beliefs_motivations": 1,
            "relationships": 1
        }
        
        # score each claim
        scored_claims = []
        for claim in claims:
            importance_score = importance_scores.get(claim.importance, 1)
            category_score = constraint_categories.get(claim.category, 1)
            
            total_score = importance_score * category_score
            scored_claims.append((total_score, claim))
        
        # sort by score descending
        scored_claims.sort(reverse=True, key=lambda x: x[0])
        
        # take top claims
        max_claims = 25
        selected = [claim for score, claim in scored_claims[:max_claims]]
        
        self._log_claim_distribution(selected)
        return selected
    
    def _log_claim_distribution(self, claims: List[Claim]) -> None:
        """Log claim distribution by category."""
        categories = {}
        importances = {}
        
        for claim in claims:
            categories[claim.category] = categories.get(claim.category, 0) + 1
            importances[claim.importance] = importances.get(claim.importance, 0) + 1
        
        logger.debug(f"Claim categories: {categories}")
        logger.debug(f"Claim importance: {importances}")
    
    def get_high_priority_claims(self, claims: List[Claim]) -> List[Claim]:
        """Filter to high-priority claims."""
        return [c for c in claims if c.importance == "high"]
    
    def group_claims_by_category(self, claims: List[Claim]) -> dict[str, List[Claim]]:
        """Group claims by category."""
        grouped = {}
        for claim in claims:
            if claim.category not in grouped:
                grouped[claim.category] = []
            grouped[claim.category].append(claim)
        
        return grouped


class ClaimValidator:
    """Validate claims are specific and testable."""
    
    @staticmethod
    def is_testable(claim: Claim) -> bool:
        """Check if claim is testable."""
        text = claim.text.lower()
        
        # flag overly vague claims
        vague_phrases = [
            "had a difficult time",
            "went through challenges",
            "experienced things",
            "lived a life",
            "was influenced by",
        ]
        
        for phrase in vague_phrases:
            if phrase in text:
                return False
        
        # must contain specific content
        if len(text.split()) < 5:
            return False
        
        return True
    
    @staticmethod
    def validate_claims(claims: List[Claim]) -> List[Claim]:
        """Filter to testable claims only."""
        valid_claims = []
        
        for claim in claims:
            if ClaimValidator.is_testable(claim):
                valid_claims.append(claim)
            else:
                logger.debug(f"Filtered vague claim: {claim.text[:50]}")
        
        logger.info(f"Validated {len(valid_claims)}/{len(claims)} claims as testable")
        return valid_claims