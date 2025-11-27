"""
Complete Data Pipeline Runner
==============================

Runs the complete data collection, sentiment analysis, and fusion pipeline.

Steps:
1. Collect news from GoScraper (filtered by stock_data dates)
2. Perform sentiment analysis on collected news
3. Fuse stock data with sentiment features

Usage:
    python3 run_data_pipeline.py
"""

import subprocess
import sys
from datetime import datetime

print("="*70)
print("COMPLETE DATA PIPELINE")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

def run_step(step_num, total_steps, script_name, description):
    """Run a pipeline step"""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ Step {step_num} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Step {step_num} failed with exit code {e.returncode}")
        return False

    except FileNotFoundError:
        print(f"\n✗ Script {script_name} not found")
        return False

# Run pipeline steps
steps = [
    ("collect_news.py", "News Collection (GoScraper)"),
    ("process_sentiment.py", "Sentiment Analysis"),
    ("fuse_data.py", "Data Fusion"),
]

total_steps = len(steps)
failed = False

for i, (script, description) in enumerate(steps, 1):
    success = run_step(i, total_steps, script, description)
    if not success:
        failed = True
        break

# Final summary
print("\n" + "="*70)
if not failed:
    print("PIPELINE COMPLETE ✓")
    print("="*70)
    print("\nGenerated files:")
    print("  1. collected_news.json      - Filtered news articles")
    print("  2. news_sentiment.json      - Sentiment analysis results")
    print("  3. fused_data/training_data.csv - Training-ready dataset")
    print("\nYou can now use fused_data/training_data.csv to train your model!")
else:
    print("PIPELINE FAILED ✗")
    print("="*70)
    print("\nPlease check the error messages above and fix the issues.")
    sys.exit(1)

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
