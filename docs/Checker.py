"""
Kharagpur Data Science Hackathon 2026
Track A: Backstory Consistency Checker using Pathway

This system evaluates whether a hypothetical character backstory
is consistent with a long-form narrative.
"""

import pathway as pw
from pathway.xpacks.llm import embedders, llms
from pathway.xpacks.llm.vector_store import VectorStoreServer
import anthropic
import os
from typing import List, Dict, Tuple
import json

class BackstoryConsistencyChecker:
    """
    Main system for checking backstory consistency against narratives.
    Uses Pathway for document management and LLM for reasoning.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the checker with API credentials."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
    def chunk_narrative(self, narrative_text: str, chunk_size: int = 3000) -> List[str]:
        """
        Intelligently chunk the narrative for processing.
        Uses paragraph boundaries to maintain context.
        """
        paragraphs = narrative_text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def extract_backstory_claims(self, backstory: str) -> List[str]:
        """
        Extract key claims from the backstory that need verification.
        Uses LLM to identify testable assertions.
        """
        prompt = f"""Analyze this character backstory and extract specific, testable claims about:
- Character traits and personality
- Key life events and experiences
- Beliefs, motivations, and fears
- Skills, abilities, or knowledge
- Relationships or social background

Backstory:
{backstory}

Return a JSON list of claims, each as a concise statement.
Example: ["Character has military training", "Character fears abandonment due to childhood trauma"]
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response
        content = response.content[0].text
        try:
            # Extract JSON from response
            start = content.find('[')
            end = content.rfind(']') + 1
            claims = json.loads(content[start:end])
            return claims
        except:
            # Fallback: split by lines
            return [line.strip() for line in content.split('\n') if line.strip()]
    
    def find_relevant_passages(self, narrative_chunks: List[str], claim: str, top_k: int = 5) -> List[str]:
        """
        Find narrative passages most relevant to a specific claim.
        Uses simple keyword matching (can be enhanced with embeddings).
        """
        # Simple scoring based on keyword overlap
        claim_words = set(claim.lower().split())
        
        scored_chunks = []
        for chunk in narrative_chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(claim_words & chunk_words)
            scored_chunks.append((overlap, chunk))
        
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def check_claim_consistency(self, claim: str, passages: List[str]) -> Tuple[bool, str]:
        """
        Check if a single claim is consistent with narrative passages.
        Returns (is_consistent, reasoning).
        """
        passages_text = "\n\n---\n\n".join(passages[:3])  # Use top 3 passages
        
        prompt = f"""You are analyzing narrative consistency.

CLAIM from character backstory:
{claim}

RELEVANT PASSAGES from the narrative:
{passages_text}

Task: Determine if this claim is CONSISTENT with the narrative evidence.

A claim is INCONSISTENT if:
- It directly contradicts stated facts
- It makes later events causally impossible
- It violates established character traits or abilities
- It creates logical contradictions

A claim is CONSISTENT if:
- It aligns with or explains character behavior
- It makes later events more plausible
- It doesn't contradict any established facts
- Evidence is neutral or supportive

Respond with:
1. VERDICT: "CONSISTENT" or "INCONSISTENT"
2. REASONING: Brief explanation (2-3 sentences)

Format:
VERDICT: [your verdict]
REASONING: [your reasoning]"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        
        # Parse verdict
        is_consistent = "CONSISTENT" in result.split("REASONING:")[0]
        reasoning = result.split("REASONING:")[-1].strip() if "REASONING:" in result else result
        
        return is_consistent, reasoning
    
    def evaluate_backstory(self, narrative_path: str, backstory: str) -> Tuple[int, str]:
        """
        Main evaluation function.
        Returns (prediction, rationale).
        prediction: 1 = consistent, 0 = inconsistent
        """
        print("Loading narrative...")
        with open(narrative_path, 'r', encoding='utf-8') as f:
            narrative = f.read()
        
        print("Chunking narrative...")
        chunks = self.chunk_narrative(narrative)
        print(f"Created {len(chunks)} chunks")
        
        print("Extracting backstory claims...")
        claims = self.extract_backstory_claims(backstory)
        print(f"Found {len(claims)} claims to verify")
        
        # Check each claim
        inconsistencies = []
        evidence = []
        
        for i, claim in enumerate(claims[:10]):  # Limit to top 10 claims for efficiency
            print(f"Checking claim {i+1}/{min(len(claims), 10)}: {claim[:50]}...")
            
            relevant_passages = self.find_relevant_passages(chunks, claim)
            is_consistent, reasoning = self.check_claim_consistency(claim, relevant_passages)
            
            evidence.append(f"Claim: {claim}\nVerdict: {'✓' if is_consistent else '✗'}\n{reasoning}")
            
            if not is_consistent:
                inconsistencies.append(claim)
        
        # Make final decision
        consistency_ratio = 1 - (len(inconsistencies) / max(len(claims[:10]), 1))
        
        # Decision threshold: if >30% of claims are inconsistent, backstory is inconsistent
        prediction = 1 if consistency_ratio > 0.7 else 0
        
        # Generate rationale
        if prediction == 0:
            rationale = f"Found {len(inconsistencies)} inconsistencies. Key issues: {inconsistencies[0] if inconsistencies else 'Multiple contradictions'}"
        else:
            rationale = f"Backstory aligns with narrative. {len(claims[:10])} claims verified with {consistency_ratio:.1%} consistency."
        
        return prediction, rationale

# Example usage
if __name__ == "__main__":
    # Initialize checker
    checker = BackstoryConsistencyChecker()
    
    # Process dataset
    results = []
    
    # Example: Process a single story
    story_id = 1
    narrative_path = "narratives/story_1.txt"  # Update with actual path
    backstory = """
    [Your backstory text here]
    """
    
    prediction, rationale = checker.evaluate_backstory(narrative_path, backstory)
    
    results.append({
        'Story ID': story_id,
        'Prediction': prediction,
        'Rationale': rationale
    })
    
    print(f"\nResults for Story {story_id}:")
    print(f"Prediction: {prediction}")
    print(f"Rationale: {rationale}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print("\nResults saved to results.csv")