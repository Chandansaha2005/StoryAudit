"""
Advanced implementation using Pathway's Vector Store.
"""

import pathway as pw
from pathway.xpacks.llm import embedders, llms, prompts
import os
from typing import List, Dict, Tuple
import json
import google.generativeai as genai

class PathwayBackstoryChecker:
    """Advanced checker using Pathway vector store."""
    
    def __init__(self, api_key: str = None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
    def setup_pathway_pipeline(self, data_dir: str):
        """Set up Pathway pipeline for document ingestion."""
        # read documents from directory
        documents = pw.io.fs.read(
            data_dir,
            format="text",
            mode="static"
        )
        
        # split into chunks
        chunked = documents.select(
            text=pw.apply(self.chunk_text, documents.data)
        )
        
        return chunked
    
    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Create overlapping chunks for context preservation."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.8:  # good break point
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def build_evaluation_system(self):
        """Build complete evaluation system for consistency."""
        system_prompt = """You are an expert literary analyst specializing in narrative consistency.

Your task is to evaluate whether character backstories are logically consistent with the events,
character development, and constraints established in a long narrative.

Key principles:
1. CAUSAL_CONSISTENCY: Does backstory make future events possible?
2. CHARACTER_CONSISTENCY: Does it align with demonstrated traits?
3. WORLD_CONSISTENCY: Does it respect narrative world rules?
4. LOGICAL_CONSISTENCY: Any logical contradictions or impossibilities?

Be rigorous but fair. Minor ambiguities should not count as inconsistencies.
Focus on fundamental contradictions that break causality or character logic."""

        return system_prompt
    
    def extract_critical_elements(self, backstory: str) -> Dict[str, List[str]]:
        """Extract structured elements from backstory text."""
        prompt = f"""Analyze this character backstory and extract elements in these categories:

1. FORMATIVE_EVENTS: Key life events that shaped the character
2. SKILLS_ABILITIES: What the character can do or knows
3. PERSONALITY_TRAITS: Core character traits and tendencies
4. RELATIONSHIPS: Important connections to other people/groups
5. BELIEFS_MOTIVATIONS: What drives the character, their worldview
6. CONSTRAINTS: Things the character cannot do or fundamental limitations

Backstory:
{backstory}

Respond in JSON format:
{{
  "formative_events": ["event1", "event2"],
  "skills_abilities": ["skill1", "skill2"],
  "personality_traits": ["trait1", "trait2"],
  "relationships": ["rel1", "rel2"],
  "beliefs_motivations": ["belief1", "belief2"],
  "constraints": ["constraint1", "constraint2"]
}}"""

        response = self.model.generate_content(prompt)
        
        content = response.text
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            elements = json.loads(content[start:end])
            return elements
        except:
            return {
                "formative_events": [],
                "skills_abilities": [],
                "personality_traits": [],
                "relationships": [],
                "beliefs_motivations": [],
                "constraints": []
            }
    
    def verify_against_narrative(self, elements: Dict[str, List[str]], 
                                 narrative_chunks: List[str]) -> Dict[str, any]:
        """
        Systematically verify each element category against the narrative.
        """
        results = {}
        
        for category, items in elements.items():
            if not items:
                continue
                
            category_results = []
            
            for item in items[:5]:  # Limit items per category
                # Find most relevant chunks
                relevant = self.find_top_chunks(item, narrative_chunks, k=3)
                
                # Verify this specific item
                verification = self.verify_single_element(
                    category, item, relevant
                )
                
                category_results.append(verification)
            
            results[category] = category_results
        
        return results
    
    def find_top_chunks(self, query: str, chunks: List[str], k: int = 3) -> List[str]:
        """
        Find most relevant chunks using keyword-based scoring.
        Can be enhanced with embeddings for production.
        """
        query_words = set(query.lower().split())
        
        scored = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            # Simple TF-IDF-like scoring
            overlap = len(query_words & chunk_words)
            score = overlap / (len(query_words) ** 0.5)
            scored.append((score, chunk))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored[:k]]
    
    def verify_single_element(self, category: str, item: str, 
                              contexts: List[str]) -> Dict[str, any]:
        """
        Verify a single backstory element against narrative evidence.
        """
        context_text = "\n\n=== PASSAGE ===\n\n".join(contexts)
        
        prompt = f"""Verify this backstory element against narrative evidence.

CATEGORY: {category}
BACKSTORY ELEMENT: {item}

NARRATIVE EVIDENCE:
{context_text}

Determine if this element is:
- SUPPORTED: Explicitly or implicitly confirmed
- NEUTRAL: Not addressed, no contradiction
- CONTRADICTED: Clearly inconsistent with evidence

Respond in format:
STATUS: [SUPPORTED/NEUTRAL/CONTRADICTED]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [brief explanation]
KEY_QUOTE: [most relevant quote from passages, if any]"""

        response = self.model.generate_content(prompt)
        
        result_text = response.text
        
        # parse result
        status = "NEUTRAL"
        if "STATUS: SUPPORTED" in result_text:
            status = "SUPPORTED"
        elif "STATUS: CONTRADICTED" in result_text:
            status = "CONTRADICTED"
        
        return {
            "item": item,
            "status": status,
            "full_result": result_text
        }
    
    def make_final_decision(self, verification_results: Dict[str, any]) -> Tuple[int, str]:
        """
        Aggregate all verification results into final decision.
        """
        contradiction_count = 0
        neutral_count = 0
        supported_count = 0
        total_items = 0
        
        contradictions = []
        
        for category, results in verification_results.items():
            for result in results:
                total_items += 1
                status = result.get('status', 'NEUTRAL')
                
                if status == 'CONTRADICTED':
                    contradiction_count += 1
                    contradictions.append(f"{category}: {result['item']}")
                elif status == 'SUPPORTED':
                    supported_count += 1
                else:
                    neutral_count += 1
        
        # Decision logic
        if total_items == 0:
            return 1, "No testable claims found in backstory"
        
        contradiction_ratio = contradiction_count / total_items
        
        # Threshold: If more than 20% contradictions, mark as inconsistent
        if contradiction_ratio > 0.2:
            prediction = 0
            rationale = f"Found {contradiction_count}/{total_items} contradictions. Key issues: {'; '.join(contradictions[:2])}"
        else:
            prediction = 1
            rationale = f"Backstory consistent: {supported_count} supported, {neutral_count} neutral, {contradiction_count} contradicted out of {total_items} elements"
        
        return prediction, rationale
    
    def evaluate(self, narrative_path: str, backstory: str) -> Tuple[int, str]:
        """
        Complete evaluation pipeline.
        """
        print("=" * 60)
        print("BACKSTORY CONSISTENCY EVALUATION")
        print("=" * 60)
        
        # Load narrative
        print("\n[1/5] Loading narrative...")
        with open(narrative_path, 'r', encoding='utf-8') as f:
            narrative = f.read()
        print(f"Loaded {len(narrative):,} characters")
        
        # Chunk narrative
        print("\n[2/5] Chunking narrative...")
        chunks = self.chunk_text(narrative)
        print(f"Created {len(chunks)} chunks")
        
        # Extract backstory elements
        print("\n[3/5] Extracting backstory elements...")
        elements = self.extract_critical_elements(backstory)
        total_elements = sum(len(v) for v in elements.values())
        print(f"Extracted {total_elements} elements across {len(elements)} categories")
        
        # Verify each element
        print("\n[4/5] Verifying against narrative...")
        verification_results = self.verify_against_narrative(elements, chunks)
        
        # Make decision
        print("\n[5/5] Making final decision...")
        prediction, rationale = self.make_final_decision(verification_results)
        
        print("\n" + "=" * 60)
        print(f"RESULT: {'CONSISTENT' if prediction == 1 else 'INCONSISTENT'}")
        print(f"RATIONALE: {rationale}")
        print("=" * 60)
        
        return prediction, rationale


# Main execution
if __name__ == "__main__":
    import sys
    
    checker = PathwayBackstoryChecker()
    
    # Example usage
    if len(sys.argv) > 2:
        narrative_path = sys.argv[1]
        backstory_path = sys.argv[2]
        
        with open(backstory_path, 'r') as f:
            backstory = f.read()
        
        prediction, rationale = checker.evaluate(narrative_path, backstory)
        
        print(f"\nPrediction: {prediction}")
        print(f"Rationale: {rationale}")
    else:
        print("Usage: python script.py <narrative_path> <backstory_path>")
