"""
config.py
System configuration and LLM prompts for KDSH Track A
"""

import os
from pathlib import Path

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the consistency checking system."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    NARRATIVES_DIR = DATA_DIR / "narratives"
    BACKSTORIES_DIR = DATA_DIR / "backstories"
    RESULTS_FILE = PROJECT_ROOT / "results.csv"
    
    # API Configuration (Google Gemini - FREE TIER)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    MODEL_NAME = "gemini-1.5-flash"
    
    # Chunking Parameters
    CHUNK_SIZE = 2500  # words per chunk
    CHUNK_OVERLAP = 300  # word overlap between chunks
    
    # Retrieval Parameters
    TOP_K_CHUNKS = 5  # number of chunks to retrieve per claim
    SIMILARITY_THRESHOLD = 0.3  # minimum similarity for retrieval
    
    # Consistency Checking
    CONTRADICTION_CONFIDENCE_THRESHOLD = 0.8  # confidence needed to mark as contradiction
    
    # LLM Parameters
    MAX_TOKENS_EXTRACTION = 3000
    MAX_TOKENS_VERIFICATION = 1500
    TEMPERATURE = 0.0  # deterministic for consistency


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

CLAIM_EXTRACTION_PROMPT = """You are analyzing a character backstory to extract testable claims.

BACKSTORY:
{backstory}

Extract specific, falsifiable claims about:
1. **Character Events**: Key life events, experiences, or history
2. **Character Traits**: Personality, behaviors, tendencies
3. **Skills/Knowledge**: What they can do, what they know
4. **Relationships**: Connections to other people or groups
5. **Beliefs/Motivations**: Worldview, goals, fears, values
6. **Physical/Biological**: Age, appearance, health conditions
7. **Constraints**: Things they CANNOT do or fundamental limitations

REQUIREMENTS:
- Each claim must be SPECIFIC and TESTABLE against narrative text
- Avoid vague statements like "had a difficult childhood"
- Prefer concrete claims like "experienced parental abandonment at age 8"
- Extract 10-25 claims total
- Focus on claims that would create narrative constraints

OUTPUT FORMAT (JSON only, no markdown):
{{
  "claims": [
    {{
      "id": "claim_1",
      "category": "character_events",
      "claim": "Character was trained in military combat before age 20",
      "importance": "high"
    }},
    {{
      "id": "claim_2",
      "category": "beliefs_motivations",
      "claim": "Character believes violence is never justified",
      "importance": "medium"
    }}
  ]
}}

IMPORTANCE levels: high (core to character), medium (significant), low (minor detail)

Return ONLY valid JSON. No preamble, no markdown formatting."""


CONSISTENCY_CHECK_PROMPT = """You are verifying whether a backstory claim is consistent with narrative evidence.

BACKSTORY CLAIM:
{claim}

NARRATIVE EVIDENCE (from the novel):
{evidence}

TASK: Determine if this claim CONTRADICTS the narrative evidence.

CONTRADICTION means:
- Direct factual contradiction (claim says X, narrative shows NOT X)
- Causal impossibility (claim makes later events impossible)
- Character trait violation (claim describes trait opposite to demonstrated behavior)
- Timeline contradiction (dates/ages don't align)
- Logical impossibility given narrative constraints

NOT a contradiction:
- Claim is not mentioned (absence of evidence ≠ contradiction)
- Minor ambiguities or interpretive differences
- Claim adds detail not covered in narrative
- Different perspectives on same event

CRITICAL: Be conservative. Only mark as CONTRADICTION if you have HIGH CONFIDENCE.

OUTPUT FORMAT (JSON only, no markdown):
{{
  "verdict": "CONSISTENT" or "CONTRADICTION",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of decision (max 2 sentences)",
  "key_evidence": "Most relevant quote from narrative evidence (if any)"
}}

Return ONLY valid JSON."""


CAUSAL_ANALYSIS_PROMPT = """You are analyzing causal relationships between backstory and narrative events.

BACKSTORY ELEMENT:
{backstory_element}

LATER NARRATIVE EVENTS:
{narrative_events}

QUESTION: Given the backstory element, are the later narrative events still CAUSALLY POSSIBLE?

Consider:
- Does backstory create conditions that PREVENT later events?
- Does backstory make character incapable of later actions?
- Does backstory establish constraints violated by later events?

OUTPUT FORMAT (JSON only):
{{
  "causally_possible": true or false,
  "confidence": 0.0 to 1.0,
  "explanation": "Why events are/aren't possible given backstory"
}}

Return ONLY valid JSON."""


# ============================================================================
# DECISION RULES
# ============================================================================

class DecisionRules:
    """Deterministic decision logic for final classification."""
    
    @staticmethod
    def should_reject(claim_results: list) -> tuple[bool, str]:
        """
        Determine if backstory should be rejected (labeled 0).
        
        Args:
            claim_results: List of consistency check results
            
        Returns:
            (should_reject: bool, reason: str)
        """
        high_confidence_contradictions = []
        medium_confidence_contradictions = []
        
        for result in claim_results:
            if result.get("verdict") == "CONTRADICTION":
                confidence = result.get("confidence", 0.0)
                claim_info = result.get("claim", "unknown")
                
                if confidence >= Config.CONTRADICTION_CONFIDENCE_THRESHOLD:
                    high_confidence_contradictions.append(claim_info)
                elif confidence >= 0.6:
                    medium_confidence_contradictions.append(claim_info)
        
        # Decision logic: ANY high-confidence contradiction → reject
        if high_confidence_contradictions:
            reason = f"High-confidence contradiction: {high_confidence_contradictions[0][:100]}"
            return True, reason
        
        # Multiple medium-confidence contradictions → reject
        if len(medium_confidence_contradictions) >= 2:
            reason = f"Multiple contradictions detected: {len(medium_confidence_contradictions)}"
            return True, reason
        
        # Otherwise accept
        reason = f"No contradictions detected ({len(claim_results)} claims verified)"
        return False, reason


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

import logging

def setup_logging(verbose: bool = False):
    """Configure logging for the system."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("pathway").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)