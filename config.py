"""
config.py
System configuration and LLM prompts for StoryAudit
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
# Use absolute path to avoid issues with different working directories
config_dir = Path(__file__).resolve().parent
env_path = config_dir / ".env"
load_dotenv(dotenv_path=env_path, override=True)

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
    
    # API Configuration (Google Gemini)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = "gemini-2.5-flash"  # Updated to use latest available model
    
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
1. Character Events - Key life events, experiences, or history
2. Character Traits - Personality, behaviors, tendencies
3. Skills/Knowledge - What they can do, what they know
4. Relationships - Connections to other people or groups
5. Beliefs/Motivations - Worldview, goals, fears, values
6. Physical/Biological - Age, appearance, health conditions
7. Constraints - Things they CANNOT do or fundamental limitations

REQUIREMENTS:
- Each claim must be SPECIFIC and TESTABLE
- Avoid vague statements like "had a difficult childhood"
- Prefer concrete claims like "was born in 1985"
- Extract 8-20 claims total
- Focus on claims that create narrative constraints

OUTPUT FORMAT - Plain text, one claim per line:
CATEGORY | CLAIM TEXT | IMPORTANCE

Examples:
character_events | Character was born in San Francisco | high
beliefs_motivations | Character believes loyalty is paramount | medium
skills_knowledge | Character is skilled in mathematics | high
character_traits | Character tends to be cautious | low
constraints | Character cannot swim | high
relationships | Character has a sibling named Marcus | medium
physical | Character is left-handed | low

Rules:
- Start each line with one of: character_events, character_traits, skills_knowledge, relationships, beliefs_motivations, physical, constraints
- Separate fields with | (pipe character)
- Use importance: high, medium, or low
- Do NOT use quotes around claims
- Do NOT use JSON format
- One claim per line only
- No preamble, just the claims"""


MULTI_STEP_VERIFICATION_PROMPT = """Perform multi-step verification of a backstory claim against narrative evidence.

BACKSTORY CLAIM:
{claim}

NARRATIVE EVIDENCE:
{evidence}

Follow these steps:
1. **Entity Extraction**: What entities (people, places, objects) are mentioned?
2. **Temporal Analysis**: What time period or sequence is implied?
3. **Causal Logic**: What causes/effects are stated or implied?
4. **Contradiction Check**: Find any explicit contradictions
5. **Consistency Score**: Overall consistency assessment

OUTPUT (JSON):
{{
  "entity_match": {{"entities_in_claim": [], "entities_in_evidence": [], "match_score": 0.0}},
  "temporal_consistency": {{"claim_timeline": "", "evidence_timeline": "", "consistent": true}},
  "causal_logic": {{"causal_chains": [], "logical_gaps": [], "plausible": true}},
  "contradictions": [],
  "overall_consistency": 0.0,
  "confidence": 0.0,
  "verdict": "CONSISTENT",
  "reasoning": ""
}}"""


CAUSAL_ANALYSIS_PROMPT = """Analyze if a backstory element makes later narrative events causally plausible.

BACKSTORY ELEMENT:
{backstory_element}

NARRATIVE EVENTS:
{narrative_events}

Determine if the backstory element provides a plausible causal basis for the narrative events.
Consider: motivations, capabilities, relationships, constraints.

OUTPUT (JSON):
{{
  "causally_possible": true,
  "causal_chain": "step1 -> step2 -> step3",
  "required_assumptions": [],
  "alternative_causes": [],
  "explanation": ""
}}"""


EVIDENCE_GRADING_PROMPT = """Grade the quality of narrative evidence for a claim.

CLAIM:
{claim}

EVIDENCE PASSAGE:
{evidence}

Rate evidence quality on these dimensions:
1. **Directness**: Does it directly address the claim? (direct/indirect/tangential)
2. **Specificity**: How concrete and specific is the evidence?
3. **Consistency**: Does it align with claim details without contradiction?
4. **Strength**: How strong is this evidence overall?

OUTPUT (JSON):
{{
  "directness": "direct",
  "specificity": 0.8,
  "consistency": 0.9,
  "overall_quality": "strong",
  "grade": "A",
  "reasoning": ""
}}"""


INCONSISTENCY_DETECTION_PROMPT = """Detect specific inconsistencies between backstory and narrative.

BACKSTORY:
{backstory}

NARRATIVE SEGMENTS:
{narrative}

Find and detail any inconsistencies:
- Timeline conflicts
- Character behavior changes unexplained by backstory
- Fact contradictions
- Capability conflicts (claims can't do something but narrative shows they can)
- Relationship inconsistencies

OUTPUT (JSON):
{{
  "inconsistencies": [
    {{
      "type": "timeline",
      "backstory_element": "",
      "narrative_element": "",
      "severity": "high",
      "description": ""
    }}
  ],
  "total_conflicts": 0,
  "severity_distribution": {{"high": 0, "medium": 0, "low": 0}},
  "overall_consistency": 0.0
}}"""


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