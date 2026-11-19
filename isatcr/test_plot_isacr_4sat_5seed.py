"""Plot completed five-seed results for four constellations and algorithms."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = (Path(__file__).resolve().parent / 'training_process_data'
                / 'v1' / 'test' / 'test' / 'isacr_2_4_d3qn_4sat_5seed')
CONSTELLATIONS = ('Walker', 'OneWeb', 'Starlink', 'Telesat')
ALGORITHMS = (
    ('ISACR_2hop', 'ISACR (2 hops)', '#789B83'),
    ('ISACR_self_elimination', 'ISACR (3 hops)', '#1f77b4'),
    ('ISACR_4hop', 'ISACR (4 hops)', '#9467bd'),
    ('D3QN+', 'D3QN+', '#C58A78'),
)
METRICS = (
    ('average_delay_seconds', 'Average delay (s)'),
    ('packet_loss_rate_percent', 'Packet loss rate (%)'),
    ('average_ending_reward', 'Average reward'),
    ('average_hop_count', 'Average hop count'),
)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def collect_results(root):
    groups = {}
    coverage = []
    for constellation in CONSTELLATIONS:
        for algorithm, _, _ in ALGORITHMS:
            rows = []
            missing = []
            for seed in range(5):
                directory = root / constellation / algorithm / f'seed{seed}'
                required = ('summary.json', 'process_result.json', 'result.txt')
                if not all((directory / name).is_file() for name in required):
                    missing.append(seed)
                    continue
                process = read_json(directory / 'process_result.json')
                if process.get('exit_code') != 0:
                    missing.append(seed)
                    continue
                steps = re.findall(r'====== step (\d+) ======',
                                   (directory / 'result.txt').read_text(encoding='utf-8'))
                if not steps or int(steps[-1]) != 300:
                    missing.append(seed)
                    continue
                row = read_json(directory / 'summary.json')
                if (row.get('constellation'), row.get('algorithm'), row.get('training_seed')) != (
                        constellation, algorithm, seed):
                    raise ValueError(f'Result identity mismatch: {directory}')
                for field, _ in METRICS:
                    if not math.isfinite(float(row[field])):
                        raise ValueError(f'Invalid {field}: {directory}')
                rows.append(row)
            groups[constellation, algorithm] = rows
            coverage.append({
                'constellation': constellation, 'algorithm': algorithm,
                'completed_seeds': len(rows), 'missing_seeds': missing,
            })
    return groups, coverage


def draw(groups, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    positions = np.arange(len(CONSTELLATIONS))
    width = 0.19
    for axis, (field, label) in zip(axes.flat, METRICS):
        for index, (algorithm, name, color) in enumerate(ALGORITHMS):
            values = []
            errors = []
            for constellation in CONSTELLATIONS:
                rows = groups[constellation, algorithm]
                samples = [float(row[field]) for row in rows]
                values.append(mean(samples) if len(samples) == 5 else np.nan)
                errors.append(stdev(samples) if len(samples) == 5 else np.nan)
            x = positions + (index - 1.5) * width
            axis.bar(x, values, width, yerr=errors, capsize=3,
                     label=name, color=color, edgecolor='black', linewidth=0.5,
                     error_kw={'elinewidth': 0.8}, zorder=3)
            for location, constellation in zip(x, CONSTELLATIONS):
                count = len(groups[constellation, algorithm])
                if count < 5:
                    axis.text(location, 0.02, f'{count}/5', rotation=90,
                              ha='center', va='bottom', fontsize=8, color=color,
                              transform=axis.get_xaxis_transform())
        axis.set_xticks(positions, CONSTELLATIONS)
        axis.set_ylabel(label)
        axis.set_ylim(0, axis.get_ylim()[1] * 1.18)
        axis.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8, ncol=2, loc='upper left')
    paths = []
    for suffix in ('png', 'pdf'):
        path = output / f'four_constellations_four_algorithms.{suffix}'
        figure.savefig(path, dpi=300)
        paths.append(path)
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    groups, coverage = collect_results(args.test_root)
    output = args.output_dir or args.test_root / 'plots_4sat_5seed'
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'error_bars': 'sample standard deviation across five seeds',
        'incomplete_groups': 'not plotted; completed seed count shown instead',
        'coverage': coverage,
        'means': [{
            'constellation': constellation, 'algorithm': algorithm,
            **{field: {'mean': mean(float(row[field]) for row in rows),
                       'std': stdev(float(row[field]) for row in rows)}
               for field, _ in METRICS},
        } for (constellation, algorithm), rows in groups.items() if len(rows) == 5],
    }
    (output / 'plot_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Completed cases: {sum(len(rows) for rows in groups.values())}/80')
    for item in coverage:
        print(f"{item['constellation']} / {item['algorithm']}: {item['completed_seeds']}/5")
    for path in draw(groups, output):
        print(path)


if __name__ == '__main__':
    main()
