import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt


results_dir = Path(sys.argv[1])
entries = []

for bmk_result_file in results_dir.rglob(f'{sys.argv[2]}_*.json'):
    *_, tp_tok, conc_tok, gpu, _ = bmk_result_file.stem.split('_')
    tp = int(tp_tok.replace('tp', ''))
    conc = int(conc_tok.replace('conc', ''))

    with open(bmk_result_file) as f:
        bmk_result = json.loads(f.readline())

    entries.append({
        'hw': gpu.split('-')[0], 'tp': tp, 'conc': conc,
        'ttft': bmk_result['median_ttft_ms'], 'tpot': bmk_result['median_tpot_ms'], 'e2el': bmk_result['median_e2el_ms'],
        'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp
    })

entries.sort(key=lambda result: (result['hw'], result['tp'], result['conc']))
gpu_types = {e['hw'] for e in entries}


summary_header = f'''\
## Benchmark Results

| Hardware | TP | Conc | TTFT (ms) | TPOT (ms) | E2EL (ms) | TPUT per GPU |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
'''
print(summary_header)
for r in entries:
    print(f"| {r['hw']} | {r['tp']} | {r['conc']} | {r['ttft']:.4f} | {r['tpot']:.4f} | {r['e2el']:.4f} | {r['tput_per_gpu']:.4f} |")


def plot_tput_vs_intvty():
    fig, ax = plt.subplots()

    if 'h100' in gpu_types:
        ax.scatter(
            [1000.0 / e['tpot'] for e in entries if e['hw'] == 'h100'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'h100'],
            label='H100', color='lightgreen'
        )
    if 'h200' in gpu_types:
        ax.scatter(
            [1000.0 / e['tpot'] for e in entries if e['hw'] == 'h200'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'h200'],
            label='H200', color='green'
        )
    if 'b200' in gpu_types:
        ax.scatter(
            [1000.0 / e['tpot'] for e in entries if e['hw'] == 'b200'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'b200'],
            label='B200', color='black'
        )
    if 'mi300x' in gpu_types:
        ax.scatter(
            [1000.0 / e['tpot'] for e in entries if e['hw'] == 'mi300x'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'mi300x'],
            label='MI300X', color='pink'
        )
    if 'mi325x' in gpu_types:
        ax.scatter(
            [1000.0 / e['tpot'] for e in entries if e['hw'] == 'mi325x'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'mi325x'],
            label='MI325X', color='red'
        )
    for entry in entries:
        ax.annotate(
            str(entry['tp']), (1000.0 / entry['tpot'], entry['tput_per_gpu']),
            textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8
        )

    ax.set_xlabel('Interactivity (tok/s/user)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='GPU Type')
    fig.tight_layout()

    fig.savefig(f'tput_vs_intvty_{sys.argv[2]}.png', bbox_inches='tight')
    plt.close(fig)
plot_tput_vs_intvty()


def plot_tput_vs_e2el():
    fig, ax = plt.subplots()

    if 'h100' in gpu_types:
        ax.scatter(
            [e['e2el'] / 1000.0 for e in entries if e['hw'] == 'h100'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'h100'],
            label='H100', color='lightgreen'
        )
    if 'h200' in gpu_types:
        ax.scatter(
            [e['e2el'] / 1000.0 for e in entries if e['hw'] == 'h200'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'h200'],
            label='H200', color='green'
        )
    if 'b200' in gpu_types:
        ax.scatter(
            [e['e2el'] / 1000.0 for e in entries if e['hw'] == 'b200'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'b200'],
            label='B200', color='black'
        )
    if 'mi300x' in gpu_types:
        ax.scatter(
            [e['e2el'] / 1000.0 for e in entries if e['hw'] == 'mi300x'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'mi300x'],
            label='MI300X', color='pink'
        )
    if 'mi325x' in gpu_types:
        ax.scatter(
            [e['e2el'] / 1000.0 for e in entries if e['hw'] == 'mi325x'], [e['tput_per_gpu'] for e in entries if e['hw'] == 'mi325x'],
            label='MI325X', color='red'
        )
    for entry in entries:
        ax.annotate(
            str(entry['tp']), (entry['e2el'] / 1000.0, entry['tput_per_gpu']),
            textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8
        )

    ax.set_xlabel('End-to-end Latency (s)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='GPU Type')
    fig.tight_layout()

    fig.savefig(f'tput_vs_e2el_{sys.argv[2]}.png', bbox_inches='tight')
    plt.close(fig)
plot_tput_vs_e2el()
