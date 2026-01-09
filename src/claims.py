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
    """Represents a single testable claim from a backstory."""
    claim_id: str
    category: str
    text: str
    importance: str  # high, medium, low
    
    def __repr__(self):
        return f"Claim({self.claim_id}: {self.text[:50]}...)"


class ClaimExtractor:
    """
    Extracts structured, testable claims from character backstories
    using LLM-based decomposition.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize claim extractor.
        
        Args:
            api_key: Google Gemini API key (defaults to Config)
        """
        self.api_key = api_key or Config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Google Gemini API key required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(Config.MODEL_NAME)
        logger.info("ClaimExtractor initialized with Gemini API")
    
    def extract_claims(self, backstory: str, story_id: str) -> List[Claim]:
        """
        Extract testable claims from backstory.
        
        Args:
            backstory: Character backstory text
            story_id: Story identifier for claim IDs
            
        Returns:
            List of Claim objects
        """
        logger.info(f"Extracting claims from backstory ({len(backstory)} chars)")
        
        # Call LLM to extract structured claims
        raw_claims = self._call_llm_extraction(backstory)
        
        # Parse and validate claims
        claims = self._parse_claims(raw_claims, story_id)
        
        # Filter and prioritize
        claims = self._prioritize_claims(claims)
        
        logger.info(f"Extracted {len(claims)} claims")
        self._log_claim_distribution(claims)
        
        return claims
    
    def _call_llm_extraction(self, backstory: str) -> dict:
        """
        Call LLM to extract claims from backstory.
        Parses plain text format: CATEGORY | CLAIM | IMPORTANCE
        
        Returns:
            Dict with extracted claims
        """
        prompt = CLAIM_EXTRACTION_PROMPT.format(backstory=backstory)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=Config.TEMPERATURE,
                    max_output_tokens=Config.MAX_TOKENS_EXTRACTION
                )
            )
            
            content = response.text.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = '\n'.join(content.split('\n')[1:-1]).strip()
            
            # Parse plain text format
            claims_list = []
            line_num = 0
            
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue  # Skip empty lines and comments
                
                # Try to parse: CATEGORY | CLAIM | IMPORTANCE
                if '|' not in line:
                    continue  # Skip lines without pipes
                
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 2:
                    continue
                
                category = parts[0].lower()
                claim_text = parts[1]
                importance = parts[2].lower() if len(parts) > 2 else "medium"
                
                # Validate category
                valid_categories = {
                    'character_events', 'character_traits', 'skills_knowledge',
                    'relationships', 'beliefs_motivations', 'physical', 'constraints'
                }
                
                if category not in valid_categories:
                    continue  # Skip invalid categories
                
                if not claim_text:
                    continue  # Skip empty claims
                
                claims_list.append({
                    "id": f"claim_{line_num}",
                    "category": category,
                    "text": claim_text,
                    "importance": importance
                })
                line_num += 1
            
            logger.info(f"Successfully extracted {len(claims_list)} claims from LLM response")
            return {"claims": claims_list}
            
        except Exception as e:
            logger.error(f"Claim extraction failed: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Return empty but valid structure
            return {"claims": []}
    
    def _parse_claims(self, raw_claims: dict, story_id: str) -> List[Claim]:
        """
        Parse raw LLM output into Claim objects.
        Works with the new plain-text format.
        
        Args:
            raw_claims: Dict from LLM response with 'claims' key
            story_id: Story identifier
            
        Returns:
            List of valid Claim objects
        """
        claims = []
        
        for raw_claim in raw_claims.get("claims", []):
            try:
                # Validate it's a dict
                if not isinstance(raw_claim, dict):
                    logger.debug(f"Skipping non-dict claim: {raw_claim}")
                    continue
                
                # Get claim text (could be 'text' or 'claim' key)
                claim_text = raw_claim.get("text") or raw_claim.get("claim", "")
                claim_text = claim_text.strip()
                
                if not claim_text:
                    continue
                
                category = raw_claim.get("category", "unknown")
                importance = raw_claim.get("importance", "medium")
                claim_id = raw_claim.get("id", f"claim_{len(claims)}")
                
                # Create Claim object
                claim = Claim(
                    claim_id=f"{story_id}_{claim_id}",
                    category=category,
                    text=claim_text,
                    importance=importance
                )
                
                claims.append(claim)
                
            except Exception as e:
                logger.warning(f"Failed to parse claim: {e}")
                continue
        
        return claims
    
    def _prioritize_claims(self, claims: List[Claim]) -> List[Claim]:
        """
        Prioritize claims by importance and category.
        Focus on claims most likely to create constraints.
        
        Args:
            claims: List of all claims
            
        Returns:
            Prioritized and potentially filtered list of claims
        """
        # Define priority scoring
        importance_scores = {"high": 3, "medium": 2, "low": 1}
        
        # Categories that create strong constraints
        constraint_categories = {
            "character_events": 3,
            "skills_knowledge": 3,
            "constraints": 3,
            "physical_biological": 2,
            "personality_traits": 2,
            "beliefs_motivations": 1,
            "relationships": 1
        }
        
        # Score each claim
        scored_claims = []
        for claim in claims:
            importance_score = importance_scores.get(claim.importance, 1)
            category_score = constraint_categories.get(claim.category, 1)
            
            total_score = importance_score * category_score
            scored_claims.append((total_score, claim))
        
        # Sort by score (descending)
        scored_claims.sort(reverse=True, key=lambda x: x[0])
        
        # Take top claims (limit to reasonable number for processing)
        max_claims = 25
        selected = [claim for score, claim in scored_claims[:max_claims]]
        
        if len(claims) > max_claims:
            logger.info(f"Prioritized to top {max_claims} claims from {len(claims)}")
        
        return selected
    
    def _log_claim_distribution(self, claims: List[Claim]):
        """Log distribution of claims by category and importance."""
        categories = {}
        importances = {}
        
        for claim in claims:
            categories[claim.category] = categories.get(claim.category, 0) + 1
            importances[claim.importance] = importances.get(claim.importance, 0) + 1
        
        logger.debug(f"Claim categories: {categories}")
        logger.debug(f"Claim importance: {importances}")
    
    def get_high_priority_claims(self, claims: List[Claim]) -> List[Claim]:
        """
        Filter to only high-priority claims.
        
        Args:
            claims: All claims
            
        Returns:
            High-priority claims only
        """
        return [c for c in claims if c.importance == "high"]
    
    def group_claims_by_category(self, claims: List[Claim]) -> dict[str, List[Claim]]:
        """
        Group claims by category for structured verification.
        
        Args:
            claims: List of claims
            
        Returns:
            Dict mapping category -> list of claims
        """
        grouped = {}
        for claim in claims:
            if claim.category not in grouped:
                grouped[claim.category] = []
            grouped[claim.category].append(claim)
        
        return grouped


class ClaimValidator:
    """
    Validates that extracted claims are specific and testable.
    """
    
    @staticmethod
    def is_testable(claim: Claim) -> bool:
        """
        Check if claim is specific enough to test against narrative.
        
        Args:
            claim: Claim to validate
            
        Returns:
            True if testable, False if too vague
        """
        text = claim.text.lower()
        
        # Flag overly vague claims
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
        
        # Must contain specific content
        if len(text.split()) < 5:  # Too short
            return False
        
        return True
    
    @staticmethod
    def validate_claims(claims: List[Claim]) -> List[Claim]:
        """
        Filter claims to only testable ones.
        
        Args:
            claims: All claims
            
        Returns:
            Valid, testable claims only
        """
        valid_claims = []
        
        for claim in claims:
            if ClaimValidator.is_testable(claim):
                valid_claims.append(claim)
            else:
                logger.debug(f"Filtered vague claim: {claim.text[:50]}")
        
        logger.info(f"Validated {len(valid_claims)}/{len(claims)} claims as testable")
        return valid_claims


class ClaimRefiner:
    """
    Refines claims to make them more specific and testable.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize claim refiner."""
        self.api_key = api_key or Config.ANTHROPIC_API_KEY
        self.client = Anthropic(api_key=self.api_key)
    
    def refine_vague_claim(self, claim: Claim, backstory_context: str) -> Claim:
        """
        Refine a vague claim to be more specific.
        
        Args:
            claim: Vague claim to refine
            backstory_context: Original backstory for context
            
        Returns:
            Refined claim (or original if refinement fails)
        """
        prompt = f"""This claim is too vague to test:
"{claim.text}"

Context from backstory:
{backstory_context[:500]}

Rewrite this claim to be:
1. Specific and concrete
2. Testable against narrative text
3. Focused on a single assertion

Return only the refined claim text, nothing else."""

        try:
            response = self.client.messages.create(
                model=Config.MODEL_NAME,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            refined_text = response.content[0].text.strip()
            
            # Create refined claim
            return Claim(
                claim_id=claim.claim_id,
                category=claim.category,
                text=refined_text,
                importance=claim.importance
            )
            
        except Exception as e:
            logger.warning(f"Failed to refine claim: {e}")
            return claim  # Return original on failure