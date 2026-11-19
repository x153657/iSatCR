"""Rebuild test CSVs from raw TXT/CSV results and draw comparison figures."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from test_4_5seed import (
    DURATION,
    FAULT_CASES,
    FAULT_LOAD_SETTING,
    LOAD_FAULT_PROBABILITY,
    LOAD_SETTINGS,
    SEEDS,
    TEST_BASE_PACKET_FREQUENCY,
    TEST_SEED,
    last_step,
    mean_rows,
    result_row,
    write_rows,
    case_trace_entries,
    parse_test_result,
)


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_TEST_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test'


def resolve_default_test_root():
    """Use the test directory unless SATELLITE_TEST_ROOT is explicit."""
    configured = os.environ.get('SATELLITE_TEST_ROOT')
    if configured:
        return Path(configured)
    return BASE_TEST_ROOT


DEFAULT_TEST_ROOT = resolve_default_test_root()
ALGORITHMS = ('D3QN', 'DDQN', 'PPO', 'ISACR_self_elimination+', 'D3QN+', 'Multi_agent')
EXTRA_ALGORITHMS = ('ICS', 'GSJF')
BASELINE_SEED = -1
BASELINE_CSV_NAMES = {
    'ICS': 'PICS_test_results.csv',
}
PLOT_ALGORITHMS = (
    'ICS', 'ISACR_self_elimination+', 'D3QN+', 'D3QN',
    'DDQN', 'PPO', 'GSJF', 'Multi_agent',
)


HATCH_LINE_WIDTH = 0.12
plt.rcParams['hatch.linewidth'] = HATCH_LINE_WIDTH
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


COLORS = {
    'Multi_agent': '#B79A75',
    'GSJF': '#789B83',
    'ISACR_self_elimination+': '#1f77b4',
    'D3QN-LC': '#9467bd',
    'D3QN': '#bcbd22',
    'DDQN': '#7f7f7f',
    'PPO': '#17becf',
    'ISACR_self_elimination': '#1f77b4',
    'D3QN+': '#C58A78',
    'D3QN-RO': '#8c564b',
    'D3QN-OH': '#8c564b',
    'ICS': '#9467BD',
    'GraphPR': '#789B83',
    'LC-SR': '#ff7f0e',
}
MARKERS = {
    'Multi_agent': 'P',
    'GSJF': '^',
    'ISACR_self_elimination+': 'o',
    'D3QN-LC': '^',
    'D3QN': '*',
    'DDQN': 'v',
    'PPO': 'D',
    'ISACR_self_elimination': 'o',
    'D3QN+': 'X',
    'D3QN-RO': 'h',
    'D3QN-OH': 'h',
    'ICS': 's',
    'GraphPR': '^',
    'LC-SR': 'P',
}
LINESTYLES = {
    'Multi_agent': (0, (1, 1)),
    'GSJF': (0, (4, 1)),
    'ISACR_self_elimination+': '-.',
    'D3QN-LC': (0, (4, 2)),
    'D3QN': '-',
    'DDQN': ':',
    'PPO': '--',
    'ISACR_self_elimination': '-.',
    'D3QN+': (0, (5, 1)),
    'D3QN-RO': (0, (2, 1)),
    'D3QN-OH': (0, (2, 1)),
    'ICS': (0, (3, 1, 1, 1)),
    'GraphPR': (0, (4, 1)),
    'LC-SR': (0, (1, 1)),
}
DISPLAY_NAMES = {
    'Multi_agent': 'GraphPR',
    'GSJF': 'GCSR',
    'ISACR_self_elimination+': 'iSatCR',
    'D3QN-LC': 'D3QN-LC',
    'D3QN': 'D3QN',
    'DDQN': 'DDQN',
    'PPO': 'PPO',
    'ISACR_self_elimination': 'iSatCR',
    'D3QN+': 'D3QN+',
    'D3QN-RO': 'D3QN-RO',
    'D3QN-OH': 'D3QN-OH',
    'ICS': 'ICS',
    'GraphPR': 'GraphPR',
    'LC-SR': 'LC-SR',
}
DELAY_COMPONENTS = (
    ('average_computation_delay_seconds', 'Computing'),
    ('average_queueing_delay_seconds', 'Queuing'),
    ('average_communication_delay_seconds', 'Communication'),
)
COMPONENT_HATCHES = ('////', '\\\\\\\\', 'xxx', '......')
HATCH_COLOR = 'black'
BAR_EDGE_COLOR = 'black'
BAR_EDGE_WIDTH = 0.15
METRICS = (
    ('average_delay_seconds', 'Average delay (s)'),
    ('packet_loss_rate_percent', 'Packet loss rate (%)'),
    ('average_hop_count', 'Average hop count'),
)


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def aggregate(rows, x_field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['algorithm'], float(row[x_field]))].append(row)
    result = defaultdict(list)
    for (algorithm, x_value), items in grouped.items():
        result[algorithm].append((x_value, items))
    return result


def numeric_values(items, field):
    values = []
    for item in items:
        value = item.get(field)
        if value in (None, '') and field != 'average_communication_delay_seconds':
            continue
        try:
            if field == 'average_communication_delay_seconds':
                value = (float(item.get('average_transmission_delay_seconds'))
                         + float(item.get('average_propagation_delay_seconds')))
            else:
                value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def plot_latency_bars(axis, grouped, available_algorithms, x_label):
    """Draw grouped stacked latency bars with one stack per delay component."""
    points_by_algorithm = {
        algorithm: sorted(grouped[algorithm])
        for algorithm in available_algorithms
    }
    x_values = sorted({
        point[0]
        for points in points_by_algorithm.values()
        for point in points
    })
    positions = np.arange(len(x_values), dtype=float)
    bar_width = 0.94 / max(len(available_algorithms), 1)
    offsets = (
        np.arange(len(available_algorithms), dtype=float)
        - (len(available_algorithms) - 1) / 2
    ) * bar_width

    for algorithm_index, algorithm in enumerate(available_algorithms):
        point_map = {point[0]: point[1] for point in points_by_algorithm[algorithm]}
        total_by_x = {
            point[0]: numeric_values(point[1], 'average_delay_seconds')
            for point in points_by_algorithm[algorithm]
        }
        total_means = np.asarray([
            mean(total_by_x[x_value]) if total_by_x.get(x_value) else np.nan
            for x_value in x_values
        ])
        bottoms = np.zeros(len(x_values), dtype=float)
        for component_index, (field, _) in enumerate(DELAY_COMPONENTS):
            heights = np.asarray([
                mean(values) if values else np.nan
                for values in (
                    numeric_values(point_map.get(x_value, []), field)
                    for x_value in x_values
                )
            ], dtype=float)
            # Draw the colored segment without an edge first.  The hatch
            # layer is deliberately below the final black outline.
            axis.bar(
                positions + offsets[algorithm_index],
                heights,
                bar_width,
                bottom=bottoms,
                color=COLORS[algorithm],
                alpha=(0.95, 0.78, 0.60, 0.42)[component_index],
                hatch=None,
                edgecolor='none',
                linewidth=0.0,
                label=(
                    DISPLAY_NAMES[algorithm]
                    if component_index == 0 else None
                ),
                zorder=2,
            )
            axis.bar(
                positions + offsets[algorithm_index],
                heights,
                bar_width,
                bottom=bottoms,
                color='none',
                hatch=(
                    COMPONENT_HATCHES[component_index]
                    if component_index < len(COMPONENT_HATCHES) else '+++'
                ),
                edgecolor=HATCH_COLOR,
                linewidth=0,
                zorder=2.1,
            )
            # A final edge-only layer makes every segment divider and the
            # outside of each stacked bar black and slightly thicker than the
            # hatch strokes.
            axis.bar(
                positions + offsets[algorithm_index],
                heights,
                bar_width,
                bottom=bottoms,
                color='none',
                hatch=None,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=2.2,
            )
            bottoms += heights

        total_errors = np.asarray([
            float(point_map[x_value][0]['average_delay_seconds_std'])
            if (len(point_map.get(x_value, [])) == 1
                and point_map[x_value][0].get('average_delay_seconds_std') not in (None, ''))
            else stdev(total_by_x[x_value])
            if len(total_by_x.get(x_value, [])) > 1 else 0.0
            for x_value in x_values
        ])
        has_error = np.isfinite(total_means) & (total_errors > 0)
        if np.any(has_error):
            axis.errorbar(
                (positions + offsets[algorithm_index])[has_error],
                total_means[has_error],
                yerr=total_errors[has_error],
                fmt='none',
                ecolor=tuple(channel * 0.8 for channel in matplotlib.colors.to_rgb(COLORS[algorithm])),
                elinewidth=0.7,
                capsize=1.1,
                capthick=0.7,
                zorder=3,
            )
        single_seed = np.asarray([
            len(point_map.get(x_value, [])) == 1
            and float(point_map[x_value][0].get('seed_count') or 1) == 1
            for x_value in x_values
        ]) & np.isfinite(bottoms)
        if np.any(single_seed):
            axis.plot(
                (positions + offsets[algorithm_index])[single_seed],
                bottoms[single_seed],
                linestyle='none', marker='_', markersize=4,
                markeredgewidth=0.8, color=COLORS[algorithm], zorder=3,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels([
        f'{value:g}' for value in x_values
    ])
    axis.set_xlabel(x_label)
    axis.set_ylabel('Average delay (s)')
    axis.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(True)
    axis.spines['right'].set_visible(True)
    axis.set_xlim(-0.5, len(x_values) - 0.5)
    return [
        Patch(
            facecolor=COLORS[algorithm],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=DISPLAY_NAMES[algorithm],
        )
        for algorithm in available_algorithms
    ], [
        Patch(
            facecolor='white',
            alpha=1.0,
            hatch=(
                COMPONENT_HATCHES[index]
                if index < len(COMPONENT_HATCHES) else '+++'
            ),
            edgecolor=HATCH_COLOR,
            linewidth=HATCH_LINE_WIDTH,
            label=label,
        )
        for index, (_, label) in enumerate(DELAY_COMPONENTS)
    ]


def plot_case(rows, x_field, x_label, output_stem, title, expected_algorithms=None,
              output_dir=DEFAULT_TEST_ROOT):
    grouped = aggregate(rows, x_field)
    available_algorithms = [
        algorithm for algorithm in (expected_algorithms or PLOT_ALGORITHMS)
        if expected_algorithms is not None or grouped[algorithm]
    ]
    outputs = []
    figure, axes = plt.subplots(
        1, 3, figsize=(15.4, 4), dpi=400,
        gridspec_kw={'width_ratios': (2, 1.2, 1.2)},
    )
    figure.subplots_adjust(left=0.045, right=0.99, bottom=0.25, top=0.80, wspace=0.30)
    suffixes = ('average_delay', 'packet_loss_rate', 'average_hop_count')
    for axis, (metric, ylabel), suffix in zip(axes, METRICS, suffixes):
        if suffix == 'average_delay':
            algorithm_handles, component_handles = plot_latency_bars(
                axis, grouped, available_algorithms, x_label,
            )
            for algorithm, handle in zip(available_algorithms, algorithm_handles):
                handle.set_label(DISPLAY_NAMES[algorithm])
            legend_handles = [
                handle for column in range(4)
                for handle in algorithm_handles[column::4]
            ]
            algorithm_legend = axis.legend(
                legend_handles,
                [handle.get_label() for handle in legend_handles],
                loc='lower left',
                bbox_to_anchor=(0.0, 1.03),
                ncol=4,
                fontsize=8,
                frameon=True,
                fancybox=True,
                facecolor='white',
                edgecolor='black',
                framealpha=1.0,
                handlelength=1.2,
                handletextpad=0.2,
                columnspacing=0.4,
                borderpad=0.2,
                borderaxespad=0,
            )
            axis.add_artist(algorithm_legend)
            component_handles = [component_handles[index] for index in (0, 2, 1)]
            delay_legend = axis.legend(
                component_handles,
                [handle.get_label() for handle in component_handles],
                loc='lower right',
                bbox_to_anchor=(1.0, 1.03),
                ncol=2,
                fontsize=8,
                frameon=True,
                fancybox=True,
                facecolor='white',
                edgecolor='black',
                framealpha=1.0,
                handlelength=1.2,
                handletextpad=0.2,
                columnspacing=0.4,
                borderpad=0.2,
                borderaxespad=0,
            )
            for legend in (algorithm_legend, delay_legend):
                legend.get_frame().set_linewidth(plt.rcParams['axes.linewidth'])
            # Center both legend boxes as one group with a small fixed gap.
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            algorithm_width = algorithm_legend.get_window_extent(renderer).width / axis.bbox.width
            delay_width = delay_legend.get_window_extent(renderer).width / axis.bbox.width
            gap = 0.02
            left = (1 - algorithm_width - delay_width - gap) / 2
            algorithm_legend.set_bbox_to_anchor((left, 1.03))
            delay_legend.set_bbox_to_anchor((left + algorithm_width + gap + delay_width, 1.03))
        else:
            line_handles = []
            for algorithm in available_algorithms:
                points = sorted(grouped[algorithm])
                x_values = [point[0] for point in points]
                y_values = []
                for _, items in points:
                    values = numeric_values(items, metric)
                    if not values:
                        values = [np.nan]
                    y_values.append(mean(values))
                line, = axis.plot(
                    x_values,
                    y_values,
                    label=DISPLAY_NAMES[algorithm],
                    color=COLORS[algorithm],
                    marker=MARKERS[algorithm],
                    linestyle=LINESTYLES[algorithm],
                    linewidth=1.8,
                    markersize=6,
                    zorder=4 if algorithm == 'ISACR_self_elimination+' else 2,
                )
                line_handles.append(line)
            axis.set_xlabel(x_label)
            axis.set_ylabel(ylabel)
            if x_field == 'fault_percent':
                fault_values = [float(row[x_field]) for row in rows]
                axis.set_xticks(np.arange(min(fault_values), max(fault_values) + 1e-9, 3))
            axis.grid(True, linestyle='--', linewidth=0.6, alpha=0.55)
            axis.set_axisbelow(True)
            # Matplotlib fills legend columns first; preserve the requested row order.
            legend_handles = [
                handle for column in range(4)
                for handle in line_handles[column::4]
            ]
            axis_legend = axis.legend(
                legend_handles,
                [handle.get_label() for handle in legend_handles],
                loc='lower center',
                bbox_to_anchor=(0.5, 1.03),
                ncol=4,
                fontsize=8,
                handlelength=1.6,
                handletextpad=0.3,
                columnspacing=0.6,
                borderpad=0.2,
                frameon=True,
                fancybox=True,
                facecolor='white',
                edgecolor='black',
                framealpha=1.0,
                borderaxespad=0.2,
            )
            axis_legend.get_frame().set_linewidth(plt.rcParams['axes.linewidth'])
    if title:
        figure.suptitle(title, y=0.995)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = [axis.get_tightbbox(renderer).transformed(figure.dpi_scale_trans.inverted())
              for axis in axes]
    bottom = min(box.y0 for box in bounds) - 0.03
    top = max(box.y1 for box in bounds) + 0.03
    for suffix, box in zip(suffixes, bounds):
        crop = Bbox.from_extents(box.x0 - 0.03, bottom, box.x1 + 0.03, top)
        for extension in ('png', 'pdf'):
            path = output_dir / f'{suffix}_{output_stem}.{extension}'
            figure.savefig(path, bbox_inches=crop, pad_inches=0)
            outputs.append(path)
    plt.close(figure)
    return outputs


def algorithm_result_root(test_root, algorithm):
    """Read every algorithm from the explicitly selected test root."""
    return test_root


def model_path_from_config(raw_path):
    config_path = raw_path.parent / 'config.json'
    if not config_path.is_file():
        return ''
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return ''
    model_path = config.get('agent', {}).get('model_path', '')
    return Path(model_path) if model_path else None


def zero_result_row(algorithm, seed, kind, case, raw_path):
    """Create an explicit zero row when a case has no complete result.txt."""
    if kind == 'load':
        load_setting = case
        fault_probability = LOAD_FAULT_PROBABILITY
        fault_percent = fault_probability * 100
        case_id = case['id']
    else:
        load_setting = FAULT_LOAD_SETTING
        fault_probability = float(case['edge_p'])
        fault_percent = fault_probability * 100
        case_id = case['id']

    row = {
        'algorithm': algorithm,
        'training_seed': seed,
        'scenario_seed': TEST_SEED,
        'target_load_tasks_per_second': load_setting['target_load_tasks_per_second'],
        'poisson_rate_s': load_setting['poisson_rate_s'],
        'poisson_rate_g': load_setting['poisson_rate_g'],
        'packet_frequency': TEST_BASE_PACKET_FREQUENCY,
        'generated_tasks': 0,
        'actual_tasks_per_second': 0,
        'reached_tasks': 0,
        'lost_tasks': 0,
        'unfinished_tasks': 0,
        'finished_tasks_percent': 0,
        'lost_generated_percent': 0,
        'packet_loss_rate_percent': 0,
        'average_delay_seconds': 0,
        'average_computation_delay_seconds': 0,
        'average_queueing_delay_seconds': 0,
        'average_propagation_delay_seconds': 0,
        'average_transmission_delay_seconds': 0,
        'average_delay_component_sum_seconds': 0,
        'average_delay_accounting_gap_seconds': 0,
        'average_hop_count': 0,
        'average_computing_percent': 0,
        'average_waiting_seconds': 0,
        'average_ending_reward': 0,
        'fault_percent': fault_percent,
        'edge_p': fault_probability,
        'edge_q': 1.0 - fault_probability,
        'test_type': kind,
        'test_case_id': case_id,
        'model_path': '',
        'raw_result_file': str(raw_path),
        'data_status': 'missing_zero',
    }
    return row


def scan_raw_results(test_root, algorithms=ALGORITHMS):
    """Parse complete result.txt files and report missing/incomplete cases."""
    load_rows = []
    fault_rows = []
    missing = []
    errors = []

    for algorithm in algorithms:
        for seed in SEEDS:
            unit_root = algorithm_result_root(test_root, algorithm) / algorithm / f'seed{seed}'
            raw_name = 'raw.txt' if algorithm == 'Multi_agent' else 'result.txt'
            for case in LOAD_SETTINGS:
                raw_path = unit_root / 'load' / case['id'] / raw_name
                if not raw_path.is_file() or last_step(raw_path) != DURATION:
                    missing.append(str(raw_path))
                    continue
                try:
                    row = read_raw_row(algorithm, seed, 'load', case, raw_path)
                    row['data_status'] = 'complete'
                    load_rows.append(row)
                except Exception as error:
                    errors.append(f'{raw_path}: {error}')

            for case in FAULT_CASES:
                raw_path = unit_root / 'fault' / case['id'] / raw_name
                if not raw_path.is_file() or last_step(raw_path) != DURATION:
                    missing.append(str(raw_path))
                    continue
                try:
                    row = read_raw_row(algorithm, seed, 'fault', case, raw_path)
                    row['data_status'] = 'complete'
                    fault_rows.append(row)
                except Exception as error:
                    errors.append(f'{raw_path}: {error}')

    load_rows.sort(key=lambda row: (
        PLOT_ALGORITHMS.index(row['algorithm']),
        int(row['training_seed']),
        float(row['target_load_tasks_per_second']),
    ))
    fault_rows.sort(key=lambda row: (
        PLOT_ALGORITHMS.index(row['algorithm']),
        int(row['training_seed']),
        float(row['fault_percent']),
    ))
    return load_rows, fault_rows, missing, errors


def read_raw_row(algorithm, seed, kind, case, raw_path):
    """Read completed logs without requiring historical model files."""
    setting = case if kind == 'load' else FAULT_LOAD_SETTING
    row = parse_test_result(raw_path, algorithm, setting, DURATION)
    workload, faults = case_trace_entries(kind, case['id'])
    for prefix, entry, count_key in (
        ('workload_trace', workload, 'task_count'),
        ('fault_trace', faults, 'event_count'),
    ):
        if (row[prefix + '_sha256'] != entry['sha256']
                or row[prefix + '_count'] != int(entry[count_key])):
            raise ValueError(f'{prefix} differs from current test contract')
    fault = LOAD_FAULT_PROBABILITY if kind == 'load' else case['edge_p']
    row.update(
        training_seed=seed, scenario_seed=TEST_SEED,
        test_type=kind, test_case_id=case['id'],
        fault_percent=fault * 100, edge_p=fault, edge_q=1 - fault,
        model_path=str(model_path_from_config(raw_path) or ''),
    )
    return row


def read_baseline_csv(test_root, algorithm, validate_source_hashes=True):
    """Read one completed centralized-baseline CSV into plot-compatible rows."""
    path = test_root / algorithm / BASELINE_CSV_NAMES[algorithm]
    if not path.is_file():
        return [], [], []
    rows = []
    errors = []
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            source_rows = csv.DictReader(stream)
            for row in source_rows:
                if row.get('test_type') not in {'load', 'fault'}:
                    errors.append(f'{path}: unsupported test_type {row.get("test_type")}')
                    continue
                try:
                    kind = row['test_type']
                    cases = LOAD_SETTINGS if kind == 'load' else FAULT_CASES
                    case = next(item for item in cases if item['id'] == row['test_case_id'])
                    setting = case if kind == 'load' else FAULT_LOAD_SETTING
                    fault = LOAD_FAULT_PROBABILITY if kind == 'load' else case['edge_p']
                    for key, expected in {**{key: setting[key] for key in
                            ('target_load_tasks_per_second', 'poisson_rate_s', 'poisson_rate_g')},
                            'scenario_seed': TEST_SEED, 'edge_p': fault,
                            'edge_q': 1 - fault}.items():
                        if abs(float(row[key]) - float(expected)) > 1e-9:
                            raise ValueError(f'{key} differs from current test contract')
                    workload, faults = case_trace_entries(kind, case['id'])
                    for prefix, entry, count_key in (('workload_trace', workload, 'task_count'),
                                                      ('fault_trace', faults, 'event_count')):
                        if (row[prefix + '_sha256'] != entry['sha256']
                                or int(float(row[prefix + '_count'])) != int(entry[count_key])):
                            raise ValueError(f'{prefix} differs from current test contract')
                    raw = Path(row['raw_result_file'])
                    if not raw.is_file() or last_step(raw) != DURATION:
                        raise ValueError('raw result is missing or incomplete')
                    parsed = parse_test_result(raw, algorithm, setting, DURATION)
                    for prefix, entry, count_key in (('workload_trace', workload, 'task_count'),
                                                      ('fault_trace', faults, 'event_count')):
                        if (parsed[prefix + '_sha256'] != entry['sha256']
                                or int(parsed[prefix + '_count']) != int(entry[count_key])):
                            raise ValueError(f'{prefix} differs in raw result')
                    config_path = test_root / algorithm / 'configs' / (raw.stem + '.json')
                    config = json.loads(config_path.read_text(encoding='utf-8'))
                    if validate_source_hashes:
                        hashes = config['test_contract']['source_and_resource_sha256']
                        if not hashes:
                            raise ValueError('source/resource hashes are missing')
                        for name, digest in hashes.items():
                            if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != digest:
                                raise ValueError(f'source/resource hash differs: {name}')
                    for key in ('generated_tasks', 'reached_tasks', 'lost_tasks',
                                'average_delay_seconds', 'average_hop_count'):
                        if abs(float(row[key]) - float(parsed[key])) > 1e-6:
                            raise ValueError(f'{key} differs between CSV and raw result')
                except (KeyError, ValueError, TypeError, StopIteration, OSError, RuntimeError) as error:
                    errors.append(f'{path}: {row.get("test_case_id")}: {error}')
                    continue
                row['algorithm'] = algorithm
                # ICS is a single, non-RL reference run.  A stable
                # synthetic seed keeps their rows compatible with the existing
                # CSV schema and with the separate plotter's integer filtering.
                row['training_seed'] = BASELINE_SEED
                row['data_status'] = 'complete'
                row['baseline_source'] = str(path)
                rows.append(row)
    except (OSError, csv.Error) as error:
        errors.append(f'{path}: {error}')
    return (
        [row for row in rows if row['test_type'] == 'load'],
        [row for row in rows if row['test_type'] == 'fault'],
        errors,
    )


def scan_d3qn_plus_results(test_root):
    """Read complete D3QN+ cases when that optional test has been run."""
    load_rows = []
    fault_rows = []
    missing = []
    errors = []
    algorithm_root = test_root / 'D3QN+'
    if not algorithm_root.is_dir():
        return load_rows, fault_rows, missing, errors

    def seed_sort_key(path):
        match = re.search(r'(\d+)$', path.name)
        return (0, int(match.group(1))) if match else (1, path.name)

    seed_dirs = sorted(
        (path for path in algorithm_root.glob('seed*') if path.is_dir()),
        key=seed_sort_key,
    )
    for seed_dir in seed_dirs:
        match = re.search(r'(\d+)$', seed_dir.name)
        if not match:
            errors.append(f'{seed_dir}: seed directory name is invalid')
            continue
        seed = int(match.group(1))
        for case in LOAD_SETTINGS:
            raw_path = seed_dir / 'load' / case['id'] / 'result.txt'
            if not raw_path.is_file() or last_step(raw_path) != DURATION:
                missing.append(str(raw_path))
                continue
            try:
                row = result_row(
                    'D3QN+', seed, 'load', case,
                    model_path_from_config(raw_path), raw_path,
                )
                row['data_status'] = 'complete'
                load_rows.append(row)
            except Exception as error:
                errors.append(f'{raw_path}: {error}')

        for case in FAULT_CASES:
            raw_path = seed_dir / 'fault' / case['id'] / 'result.txt'
            if not raw_path.is_file() or last_step(raw_path) != DURATION:
                missing.append(str(raw_path))
                continue
            try:
                row = result_row(
                    'D3QN+', seed, 'fault', float(case['edge_p']),
                    model_path_from_config(raw_path), raw_path,
                )
                row['data_status'] = 'complete'
                fault_rows.append(row)
            except Exception as error:
                errors.append(f'{raw_path}: {error}')

    load_rows.sort(key=lambda row: (int(row['training_seed']), float(row['target_load_tasks_per_second'])))
    fault_rows.sort(key=lambda row: (int(row['training_seed']), float(row['fault_percent'])))
    return load_rows, fault_rows, missing, errors


def scan_gsjf_results(test_root):
    """Read the primary GSJF one-hop results, excluding diagnostic runs."""
    load_rows, fault_rows, missing, errors = [], [], [], []
    root = test_root / 'GSJF' / 'one_hop'
    if not root.is_dir():
        return load_rows, fault_rows, missing, errors
    for kind, cases, rows in (
        ('load', LOAD_SETTINGS, load_rows), ('fault', FAULT_CASES, fault_rows),
    ):
        for case in cases:
            case_id = (case['id'] if kind == 'load'
                       else f"fault_{int(round(case['edge_p'] * 100)):02d}")
            raw_path = root / kind / case_id / 'result.txt'
            if not raw_path.is_file() or last_step(raw_path) != DURATION:
                missing.append(str(raw_path))
                continue
            try:
                setting = case if kind == 'load' else FAULT_LOAD_SETTING
                row = parse_test_result(raw_path, 'GSJF', setting, DURATION)
                workload, faults = case_trace_entries(kind, case['id'])
                for prefix, entry, count_key in (
                    ('workload_trace', workload, 'task_count'),
                    ('fault_trace', faults, 'event_count'),
                ):
                    if (row[prefix + '_sha256'] != entry['sha256']
                            or row[prefix + '_count'] != int(entry[count_key])):
                        raise ValueError(f'{prefix} differs from current test contract')
                fault = LOAD_FAULT_PROBABILITY if kind == 'load' else case['edge_p']
                row.update(
                    training_seed=BASELINE_SEED, scenario_seed=TEST_SEED,
                    test_type=kind, test_case_id=case['id'],
                    fault_percent=fault * 100, edge_p=fault, edge_q=1 - fault,
                    data_status='complete',
                )
                rows.append(row)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                errors.append(f'{raw_path}: {error}')
    return load_rows, fault_rows, missing, errors


def scan_extra_results(test_root):
    """Collect ICS and GSJF rows without zero filling."""
    load_rows, fault_rows, missing, errors = [], [], [], []
    for algorithm in BASELINE_CSV_NAMES:
        baseline_load, baseline_fault, baseline_errors = read_baseline_csv(
            test_root, algorithm, validate_source_hashes=False,
        )
        load_rows.extend(baseline_load)
        fault_rows.extend(baseline_fault)
        errors.extend(baseline_errors)
    gsjf_load, gsjf_fault, gsjf_missing, gsjf_errors = scan_gsjf_results(test_root)
    load_rows.extend(gsjf_load)
    fault_rows.extend(gsjf_fault)
    missing.extend(gsjf_missing)
    errors.extend(gsjf_errors)
    return load_rows, fault_rows, missing, errors


def draw(test_root, load_rows, fault_rows):
    """Export three aligned panels per experiment, keeping aligned heights and wider loss/hop panels."""
    outputs = []
    for kind, rows, field, label in (
        ('load', load_rows, 'target_load_tasks_per_second', 'Average tasks per second'),
        ('fault', fault_rows, 'fault_percent', 'Average link failure rate (%)'),
    ):
        if not rows:
            print(f'SKIPPED {kind} plot: no valid results')
            continue
        available = tuple(
            algorithm for algorithm in PLOT_ALGORITHMS
            if algorithm != 'Multi_agent' and any(row['algorithm'] == algorithm for row in rows)
        )
        outputs.extend(plot_case(
            rows, field, label, f"dif_{'fail' if kind == 'fault' else 'load'}",
            '',
            expected_algorithms=available, output_dir=test_root,
        ))
    return outputs


def delay_standard_deviations(rows, field):
    grouped = aggregate(rows, field)
    errors = {}
    for algorithm, points in grouped.items():
        for point, items in points:
            values = numeric_values(items, 'average_delay_seconds')
            errors[algorithm, point] = stdev(values) if len(values) > 1 else 0.0
    return errors


def plot_mean_rows(rows, field):
    result = mean_rows(rows, ('algorithm', field))
    errors = delay_standard_deviations(rows, field)
    for row in result:
        row['average_delay_seconds_std'] = errors[row['algorithm'], float(row[field])]
    return result


def read_plot_tables(output_root, require_complete=False):
    tables = []
    for kind, field, cases in (
        ('load', 'target_load_tasks_per_second', LOAD_SETTINGS),
        ('fault', 'fault_percent', FAULT_CASES),
    ):
        path = output_root / f'five_seed_{kind}_mean.csv'
        rows = read_rows(path)
        if not rows:
            raise ValueError(f'Empty plotting CSV: {path}')
        # Older mean tables omitted dispersion. Recover it from the saved seed
        # observations without overwriting edited means or scanning raw TXT.
        if any(row.get('average_delay_seconds_std') in (None, '') for row in rows):
            detail_path = output_root / f'all_{kind}_results.csv'
            if detail_path.is_file():
                errors = delay_standard_deviations(read_rows(detail_path), field)
                for row in rows:
                    if row.get('average_delay_seconds_std') in (None, ''):
                        key = row['algorithm'], float(row[field])
                        if key in errors:
                            row['average_delay_seconds_std'] = errors[key]
        keys = set()
        for row in rows:
            key = row['algorithm'], float(row[field])
            if key in keys:
                raise ValueError(f'Duplicate plotting point {key}: {path}')
            keys.add(key)
            for column in (field, *(metric for metric, _ in METRICS),
                           'average_computation_delay_seconds', 'average_queueing_delay_seconds',
                           'average_transmission_delay_seconds', 'average_propagation_delay_seconds'):
                if not math.isfinite(float(row[column])):
                    raise ValueError(f'Invalid {column} at {key}: {path}')
            if row.get('average_delay_seconds_std') not in (None, ''):
                error = float(row['average_delay_seconds_std'])
                if not math.isfinite(error) or error < 0:
                    raise ValueError(f'Invalid delay standard deviation at {key}: {path}')
        if require_complete:
            expected_points = {
                float(case[field]) if kind == 'load' else float(case['edge_p']) * 100
                for case in cases
            }
            for algorithm in PLOT_ALGORITHMS:
                selected = [row for row in rows if row['algorithm'] == algorithm]
                seeds = 1 if algorithm in EXTRA_ALGORITHMS else len(SEEDS)
                if ({float(row[field]) for row in selected} != expected_points
                        or any(int(row['seed_count']) != seeds for row in selected)):
                    raise ValueError(f'Incomplete {algorithm} plotting CSV: {path}')
        print(f'Plot CSV: {path}; {len(rows)} rows')
        tables.append(rows)
    return tables


def build_coverage(load_rows, fault_rows):
    coverage = []
    for kind, cases, rows in (('load', LOAD_SETTINGS, load_rows), ('fault', FAULT_CASES, fault_rows)):
        for algorithm in PLOT_ALGORITHMS:
            expected_seeds = 1 if algorithm in EXTRA_ALGORITHMS else len(SEEDS)
            for case in cases:
                seeds = sorted({int(row['training_seed']) for row in rows if row['algorithm'] == algorithm and row['test_case_id'] == case['id']})
                coverage.append({'algorithm': algorithm, 'test_type': kind, 'test_case_id': case['id'], 'completed_seeds': len(seeds), 'expected_seeds': expected_seeds, 'seeds': ','.join(map(str, seeds))})
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root', type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument('--output-dir', type=Path, help='Plot/CSV output directory; defaults to <test-root>/plots_4_5seed.')
    parser.add_argument('--refresh-csv', action='store_true', help='Explicitly rebuild and overwrite CSVs from raw results.')
    parser.add_argument(
        '--require-complete', action='store_true',
        help='Refuse to write output unless all expected TXT cases are present.',
    )
    parser.add_argument(
        '--status-only', action='store_true',
        help='Only scan TXT files; do not write CSVs or figures.',
    )
    args = parser.parse_args()
    test_root = args.test_root.resolve()
    output_root = (args.output_dir or test_root / 'plots_4_5seed').resolve()
    table_paths = [output_root / f'five_seed_{kind}_mean.csv' for kind in ('load', 'fault')]
    if not args.status_only and not args.refresh_csv and all(path.is_file() for path in table_paths):
        load_rows, fault_rows = read_plot_tables(output_root, args.require_complete)
        for output in draw(output_root, load_rows, fault_rows):
            print(output)
        return
    print(f'Test result root: {test_root}')
    for algorithm in PLOT_ALGORITHMS:
        print(f'Source {algorithm}: {algorithm_result_root(test_root, algorithm) / algorithm}')
    load_rows, fault_rows, missing, errors = scan_raw_results(test_root)
    extra_load_rows, extra_fault_rows, extra_missing, extra_errors = scan_extra_results(test_root)
    load_rows.extend(extra_load_rows)
    fault_rows.extend(extra_fault_rows)
    missing.extend(extra_missing)
    errors.extend(extra_errors)
    coverage = build_coverage(load_rows, fault_rows)
    for algorithm in PLOT_ALGORITHMS:
        entries = [row for row in coverage if row['algorithm'] == algorithm]
        found = sum(row['completed_seeds'] for row in entries)
        total = sum(row['expected_seeds'] for row in entries)
        state = 'SKIPPED: no valid results' if not found else ('complete' if found == total else 'PARTIAL: available results only')
        print(f'{algorithm}: {found}/{total}; {state}')
    expected = len(ALGORITHMS) * len(SEEDS) * (
        len(LOAD_SETTINGS) + len(FAULT_CASES)
    )
    base_missing = len(missing) - len(extra_missing)
    complete_found = expected - base_missing
    base_found = len(load_rows) + len(fault_rows) - len(extra_load_rows) - len(extra_fault_rows)
    print(f'Complete training TXT cases: {complete_found}/{expected}; valid rows collected: {base_found}/{expected}')
    if extra_load_rows or extra_fault_rows:
        extra_counts = {
            algorithm: sum(
                row['algorithm'] == algorithm
                for row in extra_load_rows + extra_fault_rows
            )
            for algorithm in EXTRA_ALGORITHMS
        }
        print(f'Additional algorithm rows: {extra_counts}')
    if extra_missing:
        print(f'Missing/incomplete optional cases: {len(extra_missing)}')
    if missing:
        print(f'Missing/incomplete TXT cases: {len(missing)}')
    if errors:
        print(f'Parse errors: {len(errors)}')
        for error in errors[:10]:
            print(error)
    if args.status_only:
        return
    if (any(row['completed_seeds'] != row['expected_seeds'] for row in coverage) or errors) and args.require_complete:
        raise SystemExit(
            'Incomplete data; finish the missing result files or omit '
            '--require-complete to use the available cases.'
        )
    if not load_rows and not fault_rows:
        raise SystemExit('No valid complete results were found for any algorithm.')

    output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ('coverage.csv', coverage), ('all_load_results.csv', load_rows),
        ('all_fault_results.csv', fault_rows),
    ):
        path = output_root / name
        if args.refresh_csv or not path.is_file():
            write_rows(path, rows)
    for path, rows, field in (
        (table_paths[0], load_rows, 'target_load_tasks_per_second'),
        (table_paths[1], fault_rows, 'fault_percent'),
    ):
        if args.refresh_csv or not path.is_file():
            write_rows(path, plot_mean_rows(rows, field))
    load_rows, fault_rows = read_plot_tables(output_root, args.require_complete)
    for output in draw(output_root, load_rows, fault_rows):
        print(output)


if __name__ == '__main__':
    main()
