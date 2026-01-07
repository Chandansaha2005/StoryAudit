#!/usr/bin/env python
"""
demo_pipeline.py
Demonstrates the StoryAudit pipeline workflow without requiring API credentials.
This shows how each component works and that the project is correctly structured.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_load_documents():
    """Demonstrate document loading."""
    print("\n" + "="*60)
    print("STAGE 1: DOCUMENT LOADING")
    print("="*60)
    
    from ingest import NarrativeLoader, BackstoryLoader
    from config import Config
    
    narratives_dir = Config.NARRATIVES_DIR
    backstories_dir = Config.BACKSTORIES_DIR
    
    print(f"\nNarratives directory: {narratives_dir}")
    print(f"Backstories directory: {backstories_dir}")
    
    # List available files
    narrative_files = list(narratives_dir.glob("*.txt"))
    backstory_files = list(backstories_dir.glob("*.txt"))
    
    print(f"\nFound {len(narrative_files)} narrative file(s):")
    for f in narrative_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\nFound {len(backstory_files)} backstory file(s):")
    for f in backstory_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    if narrative_files and backstory_files:
        print("\n✓ Document loading would work correctly")
        
        # Try to load story 1
        try:
            loader = NarrativeLoader(narratives_dir)
            narrative = loader.load_narrative("1")
            print(f"✓ Successfully loaded narrative for story 1")
            print(f"  - Content length: {len(narrative):,} characters")
            print(f"  - Word count: {len(narrative.split()):,} words")
            
            metadata = loader.get_narrative_metadata(narrative)
            print(f"  - Metadata: {metadata}")
            
        except Exception as e:
            print(f"✗ Failed to load narrative: {e}")
        
        try:
            loader = BackstoryLoader(backstories_dir)
            backstory = loader.load_backstory("1")
            print(f"\n✓ Successfully loaded backstory for story 1")
            print(f"  - Content length: {len(backstory):,} characters")
            print(f"  - Word count: {len(backstory.split()):,} words")
            
        except Exception as e:
            print(f"✗ Failed to load backstory: {e}")
    else:
        print("\n✗ Missing data files")


def demo_chunking():
    """Demonstrate narrative chunking."""
    print("\n" + "="*60)
    print("STAGE 2: NARRATIVE CHUNKING")
    print("="*60)
    
    from chunk import NarrativeChunker, ChunkIndex
    from ingest import NarrativeLoader
    from config import Config
    
    try:
        loader = NarrativeLoader(Config.NARRATIVES_DIR)
        narrative = loader.load_narrative("1")
        
        chunker = NarrativeChunker(
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP
        )
        
        print(f"\nChunking with parameters:")
        print(f"  - Chunk size: {Config.CHUNK_SIZE} words")
        print(f"  - Overlap: {Config.CHUNK_OVERLAP} words")
        
        chunks = chunker.chunk_narrative(narrative, "story_1")
        
        print(f"\n✓ Successfully chunked narrative")
        print(f"  - Total chunks: {len(chunks)}")
        
        if chunks:
            print(f"  - First chunk: {len(chunks[0].text)} chars, {chunks[0].word_count} words")
            print(f"  - Last chunk: {len(chunks[-1].text)} chars, {chunks[-1].word_count} words")
        
        # Create index
        index = ChunkIndex(chunks)
        print(f"\n✓ Created chunk index")
        print(f"  - Chunks in order: {len(index.get_chunks_in_order())}")
        
        # Test retrieval
        sample_chunk = index.get_chunk_by_order(0)
        print(f"  - Retrieved chunk 0: {sample_chunk.chunk_id}")
        
    except Exception as e:
        print(f"✗ Chunking failed: {e}")


def demo_claims():
    """Demonstrate claim extraction (without API)."""
    print("\n" + "="*60)
    print("STAGE 3: CLAIM EXTRACTION (SIMULATED)")
    print("="*60)
    
    from claims import Claim
    
    print(f"\nClaim structure:")
    print(f"  - ID: Unique identifier")
    print(f"  - Category: Type of claim (events, traits, skills, etc.)")
    print(f"  - Text: The actual claim statement")
    print(f"  - Importance: high/medium/low")
    
    # Show example claims
    example_claims = [
        Claim("story_1_claim_001", "character_events", 
              "Character was trained in military combat before age 20", "high"),
        Claim("story_1_claim_002", "skills_knowledge",
              "Character speaks at least 3 languages fluently", "high"),
        Claim("story_1_claim_003", "personality_traits",
              "Character is primarily driven by ambition rather than compassion", "medium"),
    ]
    
    print(f"\nExample extracted claims:")
    for claim in example_claims:
        print(f"  [{claim.importance.upper()}] {claim.text}")
        print(f"           Category: {claim.category}")
    
    print(f"\n✓ Claim structure is properly defined")
    print(f"  - In real execution, {len(example_claims)} example claims would be extracted using Gemini API")


def demo_retrieval():
    """Demonstrate evidence retrieval."""
    print("\n" + "="*60)
    print("STAGE 4: EVIDENCE RETRIEVAL")
    print("="*60)
    
    from retrieve import EvidenceRetriever, PathwayEvidenceIndex
    from chunk import NarrativeChunker, ChunkIndex
    from claims import Claim
    from ingest import NarrativeLoader
    from config import Config
    
    try:
        loader = NarrativeLoader(Config.NARRATIVES_DIR)
        narrative = loader.load_narrative("1")
        
        chunker = NarrativeChunker(
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP
        )
        
        chunks = chunker.chunk_narrative(narrative, "story_1")
        chunk_index = ChunkIndex(chunks)
        
        # Create retriever
        retriever = EvidenceRetriever(chunk_index)
        
        print(f"\n✓ Created evidence retriever")
        print(f"  - Initialized with {len(chunks)} chunks")
        
        # Test retrieval with a sample claim
        sample_claim = Claim(
            "test_claim",
            "character_events",
            "Character traveled to the Eastern Provinces",
            "high"
        )
        
        evidence = retriever.retrieve_evidence(sample_claim, top_k=Config.TOP_K_CHUNKS)
        
        print(f"\n✓ Retrieved evidence for sample claim")
        print(f"  - Claim: '{sample_claim.text}'")
        print(f"  - Found {len(evidence)} relevant chunks")
        
        if evidence:
            for i, chunk in enumerate(evidence):
                print(f"    Chunk {i+1} (order {chunk.temporal_order}): "
                      f"{chunk.word_count} words")
        
        # Test Pathway evidence index
        print(f"\n✓ Pathway integration:")
        pathway_index = PathwayEvidenceIndex(chunk_index)
        print(f"  - Built inverted index with terms from chunks")
        print(f"  - Ready for fast evidence retrieval")
        
    except Exception as e:
        print(f"✗ Retrieval failed: {e}")


def demo_verification():
    """Demonstrate verification structure (without API)."""
    print("\n" + "="*60)
    print("STAGE 5: VERIFICATION (STRUCTURE ONLY)")
    print("="*60)
    
    from judge import VerificationResult, DecisionAggregator
    from claims import Claim
    from chunk import Chunk
    
    print(f"\nVerification result structure:")
    print(f"  - Verdict: CONSISTENT or CONTRADICTION")
    print(f"  - Confidence: 0.0 to 1.0")
    print(f"  - Reasoning: Explanation of decision")
    
    # Show example results
    example_results = [
        VerificationResult(
            Claim("c1", "events", "Character was born in 1626", "high"),
            "CONSISTENT",
            0.92,
            "Narrative confirms character was born in 1626",
            [],
            "In his first chapter, the narrative states Marcus was born in 1626"
        ),
        VerificationResult(
            Claim("c2", "skills", "Character speaks five languages", "medium"),
            "CONSISTENT",
            0.75,
            "Narrative mentions three languages; assumes fifth through context",
            [],
            "Narrative confirms fluency in Latin, French, and Eastern dialect"
        ),
    ]
    
    print(f"\nExample verification results:")
    for result in example_results:
        status = "✓" if result.verdict == "CONSISTENT" else "✗"
        print(f"  {status} {result.claim.text}")
        print(f"     Verdict: {result.verdict} (confidence: {result.confidence:.2f})")
        print(f"     Reasoning: {result.reasoning}")
    
    # Show decision logic
    print(f"\n✓ Decision aggregation logic:")
    print(f"  - Rule: If ANY high-confidence contradiction → label = 0")
    print(f"  - Else: label = 1")
    
    # Test with example results
    prediction, rationale = DecisionAggregator.make_decision(example_results)
    print(f"\n  Example decision:")
    print(f"    Prediction: {prediction}")
    print(f"    Rationale: {rationale}")


def demo_pipeline():
    """Demonstrate pipeline orchestration."""
    print("\n" + "="*60)
    print("STAGE 6: PIPELINE ORCHESTRATION")
    print("="*60)
    
    from pipeline import PipelineFactory
    from config import Config
    
    print(f"\nPipeline components:")
    print(f"  - Narrative Loader: Load novels from disk")
    print(f"  - Narrative Chunker: Split narrative into ordered chunks")
    print(f"  - Claim Extractor: Extract testable claims using Gemini")
    print(f"  - Evidence Retriever: Find relevant chunks for each claim")
    print(f"  - Consistency Judge: Verify claims against evidence using Gemini")
    print(f"  - Decision Aggregator: Make final consistent/inconsistent decision")
    
    # Show factory
    print(f"\n✓ Pipeline factory ready:")
    print(f"  - create_standard_pipeline(): Standard execution")
    print(f"  - create_pathway_pipeline(): Enhanced Pathway integration")
    
    print(f"\n✓ Configuration loaded:")
    print(f"  - LLM: {Config.MODEL_NAME}")
    print(f"  - API Key configured: {'Yes' if Config.GOOGLE_API_KEY else 'No'}")
    print(f"  - Chunk size: {Config.CHUNK_SIZE} words")
    print(f"  - Top-K retrieval: {Config.TOP_K_CHUNKS} chunks")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("StoryAudit DEMO: Complete Pipeline Walkthrough")
    print("="*70)
    print("\nThis demo shows each stage of the consistency checking pipeline.")
    print("In a real run, stages 3 (Claim Extraction) and 5 (Verification)")
    print("would use the Google Gemini API.")
    
    try:
        demo_load_documents()
    except Exception as e:
        print(f"Document loading demo error: {e}")
    
    try:
        demo_chunking()
    except Exception as e:
        print(f"Chunking demo error: {e}")
    
    try:
        demo_claims()
    except Exception as e:
        print(f"Claims demo error: {e}")
    
    try:
        demo_retrieval()
    except Exception as e:
        print(f"Retrieval demo error: {e}")
    
    try:
        demo_verification()
    except Exception as e:
        print(f"Verification demo error: {e}")
    
    try:
        demo_pipeline()
    except Exception as e:
        print(f"Pipeline demo error: {e}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nTo run the full system with Gemini API:")
    print("  1. Set GOOGLE_API_KEY environment variable")
    print("  2. pip install google-generativeai pathway pandas numpy")
    print("  3. python run.py --story-id 1")
    print("\nSee SETUP.md for detailed instructions.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
