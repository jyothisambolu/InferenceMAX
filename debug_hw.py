#!/usr/bin/env python3
import json
from pathlib import Path

# Test what hw values are actually in the results
results_dir = Path("results")
if results_dir.exists():
    print("Found results directory")
    for result_path in results_dir.rglob("*.json"):
        print(f"\nFile: {result_path}")
        with open(result_path) as f:
            result = json.load(f)
        print(f"  hw: '{result.get('hw', 'MISSING')}'")
        print(f"  framework: '{result.get('framework', 'MISSING')}'")
        print(f"  precision: '{result.get('precision', 'MISSING')}'")
        print(f"  model: '{result.get('model', 'MISSING')}'")
else:
    print("No results directory found")
