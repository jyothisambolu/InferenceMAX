import sys
import json
from pathlib import Path


summary_header = f'''\
| Hardware | TP | Conc | TTFT (ms) | TPOT (ms) | E2EL (s) | TPUT per GPU |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
'''
print(summary_header)

results = []
results_dir = Path(sys.argv[1])
for result_path in results_dir.rglob(f'agg_{sys.argv[2]}_*.json'):
    with open(result_path) as f:
        result = json.load(f)
    results.append(result)

results.sort(key=lambda r: (r['hw'], r['tp'], r['conc']))
for result in results:
    print(
        f"| {result['hw'].upper()} "
        f"| {result['tp']} "
        f"| {result['conc']} "
        f"| {(result['median_ttft'] * 1000):.4f} "
        f"| {(result['median_tpot'] * 1000):.4f} "
        f"| {result['median_e2el']:.4f} "
        f"| {result['tput_per_gpu']:.4f} |"
    )
