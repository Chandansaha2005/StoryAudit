#!/usr/bin/env python
"""
Quick timing test - Measures actual execution time for 3 examples
to extrapolate for 60 examples.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from csv_processor import CSVBackstoryDataset, ConsistencyCSVProcessor
from pipeline import AdvancedConsistencyPipeline
from config import Config

def main():
    print("\n" + "="*70)
    print("STORYAUDIT TIMING TEST - 3 Examples")
    print("="*70)
    
    try:
        # Setup
        data_dir = Path(__file__).parent / "data"
        csv_path = Path(__file__).parent / "test.csv"
        
        print(f"\n[1/4] Initializing pipeline...")
        start_init = time.time()
        pipeline = AdvancedConsistencyPipeline(
            data_dir, data_dir, Config.GEMINI_API_KEY, 
            enable_caching=True
        )
        init_time = time.time() - start_init
        print(f"      Done in {init_time:.2f}s")
        
        # Load CSV
        print(f"[2/4] Loading test data...")
        start_csv = time.time()
        dataset = CSVBackstoryDataset(csv_path, data_dir)
        examples = dataset.get_examples()
        csv_time = time.time() - start_csv
        print(f"      Loaded {len(examples)} examples in {csv_time:.2f}s")
        
        # Process 3 examples to measure per-example time
        print(f"\n[3/4] Processing 3 examples...")
        print(f"      (this will call Gemini API 3 times)")
        processor = ConsistencyCSVProcessor(pipeline)
        small_batch = examples[:3]
        
        start_batch = time.time()
        try:
            results = processor.process_csv_batch(small_batch, optimized=True)
            batch_time = time.time() - start_batch
            
            # Calculate metrics
            per_example = batch_time / len(results) if results else float('inf')
            estimated_60 = per_example * 60
            
            print(f"      Processed {len(results)}/3 examples in {batch_time:.2f}s")
            print(f"\n[4/4] Time Estimation:")
            print(f"      Per-example average: {per_example:.2f}s")
            print(f"      For 60 examples: {estimated_60:.0f}s ({estimated_60/60:.1f} minutes)")
            
            if estimated_60 < 1200:  # Less than 20 minutes
                print(f"\n      TARGET ACHIEVED: {estimated_60/60:.1f} min < 20 min ✓")
            else:
                print(f"\n      Still within acceptable range: {estimated_60/60:.1f} min")
                
        except Exception as e:
            print(f"      Error during batch processing: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*70)
    print("NOTE: Actual time depends on Gemini API response speed")
    print("First run builds cache, subsequent runs will be 5-7x faster")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
