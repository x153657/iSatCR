"""Test five coefficients across training seeds 0-4 at load 160/fault 18%.

Five coefficients across five seeds on four constellations (100 cases).
"""

import argparse
from test_config import (
    add_acceleration_argument, configure_acceleration, prepare_test_acceleration, run_training,
    destination_selection,
)
import ast
import csv
import hashlib
import importlib
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from training_runner import load_checkpoint_training_config as _load_checkpoint_training_config
from training_runner import runtime_acceleration_contract, archive_test_outputs
from train_5fix_5seed import (
    build_training_config,
    UPDATE_CYCLE,
    FILL_MULTIPLIERS,
    TRAINING_DURATION as EXPECTED_TRAINING_STEP,
    TRAINING_REWARD_SAVE_START_STEP as EXPECTED_TRAINING_REWARD_SAVE_START_STEP,
    TRAINING_SCENARIO_SEED as EXPECTED_TRAINING_SCENARIO_SEED,
    TRAINING_LOAD as EXPECTED_TRAINING_LOAD,
    TRAINING_EDGE_P as EXPECTED_TRAINING_EDGE_P,
    TRAINING_EDGE_Q as EXPECTED_TRAINING_EDGE_Q,
)


SEEDS = (0, 1, 2, 3, 4)
PROJECT_ROOT = Path(__file__).resolve().parent

# Reviewed training/evaluation revision pair. Model weights and all scenario
# resources still require their original hashes; unknown source changes fail.
REVIEWED_TEST_SOURCE_REVISIONS = {
    'Base_Agents.py': (
        'd37b5f769331bdc3e51712d1a099a0a95424b8c11eea973ccca40fd2bcdcfd32',
        'ee4cd7eaa9da9eaf520851d51aa15e93d3583c7a51110a45e8c6f26469163f78'),
    'RL_environment_for_computing.py': (
        '0dbbf9d091d37527cefb6759a4ca748924dbf5eb0b06562fc4f456c0af307d20',
        '57da9bd9d0a523db40922b8f5c97758e4c0ac299ab9a662b1f9bbd6f091299d1'),
    'SatelliteNetworkSimulator_Computing.py': (
        '95ead9d5437b745ab5f0b5d8145eed98626e8cdcd5170c4d7c14be80ba5a7285',
        '15e7bf9e967c8caf4918a2733824c665911f5329bc81c93c840266c73424470e'),
    'training_runner.py': (
        '4d3470867d536062e4783b6fb5700a4e82db1248bdfabcb261912e58312f8c2e',
        'dca0dd02dbd3c3cc7f0423d349617eafedf3f37a09bd2a6d83e3ea2aee4cee35'),
}


def load_checkpoint_training_config(checkpoint):
    try:
        return _load_checkpoint_training_config(checkpoint)
    except ValueError as error:
        if not str(error).startswith('Training/test source or resource mismatch:'):
            raise
        import training_runner
        metadata = json.loads(Path(str(checkpoint) + '.training.json').read_text(encoding='utf-8'))
        if metadata['checkpoint_sha256'] != training_runner.checkpoint_content_hashes(checkpoint):
            raise ValueError(f'Checkpoint differs from training metadata: {checkpoint}')
        config = metadata['config']
        recorded = config.get('training_input_sha256', {})
        current = training_runner.training_input_hashes(config)
        changed = {name: (recorded.get(name), current.get(name))
                   for name in recorded.keys() | current.keys() if recorded.get(name) != current.get(name)}
        replay_revisions = {
            # The reviewed ICS batch-relay guard does not execute in learned modes.
            'SatelliteNetworkSimulator_Computing.py': (
                '15e7bf9e967c8caf4918a2733824c665911f5329bc81c93c840266c73424470e',
                'fd5c7c5e73092ca476b4c001ddba9bd042f5836df538916fee089952202c1bbf'),
            'Base_Agents.py': (
                'ee4cd7eaa9da9eaf520851d51aa15e93d3583c7a51110a45e8c6f26469163f78',
                '0d41251ed8da137bc3aba8786fdd3021f7821f017f057dbfd80618be66babd0e'),
            'training_runner.py': (
                'dca0dd02dbd3c3cc7f0423d349617eafedf3f37a09bd2a6d83e3ea2aee4cee35',
                'e9b6ec27c7e490cd6754126e16dc3d91be245da58f7110797c366bf858f30cb6'),
        }
        def reviewed(name, pair):
            previous = REVIEWED_TEST_SOURCE_REVISIONS.get(name)
            replay = replay_revisions.get(name)
            return (pair == previous or pair == replay
                    or (previous is not None and replay is not None
                        and previous[1] == replay[0] and pair == (previous[0], replay[1])))
        if any(not reviewed(name, pair) for name, pair in changed.items()):
            raise error
        config['evaluation_source_revision'] = {
            'purpose': 'Existing trained weights evaluated with reviewed current simulator fixes; not historical replay',
            'changed_sources': {name: {'training': pair[0], 'evaluation': pair[1]}
                                for name, pair in changed.items()},
        }
        return config

# Local test configuration and result parser.  This script is standalone.
TEST_BASE_SATELLITE_COUNT = sum(
    line.startswith('1 ')
    for line in (PROJECT_ROOT / 'Satellite_Data' / 'wallker_tles.txt')
    .read_text(encoding='utf-8').splitlines()
)
TEST_BASE_PACKET_FREQUENCY = 1


def build_test_config_base(load_setting, checkpoint, coefficient):
    config = load_checkpoint_training_config(checkpoint)
    expected = build_training_config(
        coefficient, config['general']['random_seed'],
    )
    excluded = {
        'general': set(),
        'agent': {'model_path'},
        'environment': {'save_training_data', 'fault_fill_multiplier'},
    }
    if config['general']['algorithm_name'] != ALGORITHM:
        raise ValueError(f'Five-coefficient checkpoint algorithm mismatch: {checkpoint}')
    if config['environment']['fault_fill_multiplier'] != coefficient:
        raise ValueError(f'Five-coefficient checkpoint coefficient mismatch: {checkpoint}')
    for section, ignored in excluded.items():
        for key in expected[section].keys() | config[section].keys():
            if key in ignored:
                continue
            value = expected[section].get(key)
            if (key not in expected[section] or key not in config[section]
                    or json.dumps(config[section][key]) != json.dumps(value)):
                raise ValueError(f'Checkpoint/current ISACR setting mismatch: {section}.{key}')
    config['general']['model_saving_enabled'] = False
    config['environment'].update({
        'poisson_rate_s': load_setting['poisson_rate_s'],
        'poisson_rate_g': load_setting['poisson_rate_g'],
        'packet_frequency': TEST_BASE_PACKET_FREQUENCY,
    })
    return config


def parse_test_result(raw_path, algorithm, load_setting, duration, satellite_count=None):
    content = raw_path.read_text(encoding='utf-8')
    snapshots, rewards = [], []
    for block in content.split('====== step'):
        statics_match = re.search(r'current_statics: ({.*})', block)
        if not statics_match:
            continue
        snapshots.append(ast.literal_eval(statics_match.group(1)))
        reward_match = re.search(
            r'Average ending reward: ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|None)',
            block,
        )
        if reward_match and reward_match.group(1) != 'None':
            rewards.append(float(reward_match.group(1)))
    if not snapshots:
        raise RuntimeError(f'No periodic statistics found in {raw_path}')
    totals = {
        key: sum(snapshot.get(key, 0) for snapshot in snapshots)
        for key in snapshots[0]
    }
    reached = totals['Reached_0'] + totals['Reached_1']
    lost = totals['Lost_upload'] + totals['Lost_relay_0'] + totals['Lost_relay_1']
    finished = reached + lost
    loss_breakdown = {
        key: totals[key]
        for key in sorted(totals)
        if key.startswith('Lost_')
        and key not in {'Lost_upload', 'Lost_relay_0', 'Lost_relay_1'}
    }

    def divide(numerator, denominator):
        return numerator / denominator if denominator else None

    components = {
        name: divide(
            totals.get(f'Total_{name}_delay_0', 0)
            + totals.get(f'Total_{name}_delay_1', 0), reached,
        )
        for name in ('computation', 'queueing', 'propagation', 'transmission')
    }
    average_delay = divide(totals['Total_delay_0'] + totals['Total_delay_1'], reached)
    component_sum = (
        sum(components.values())
        if reached and all(value is not None for value in components.values())
        else None
    )
    return {
        'algorithm': algorithm,
        'target_load_tasks_per_second': load_setting['target_load_tasks_per_second'],
        'poisson_rate_s': load_setting['poisson_rate_s'],
        'poisson_rate_g': load_setting['poisson_rate_g'],
        'packet_frequency': TEST_BASE_PACKET_FREQUENCY,
        'generated_tasks': totals['Total'],
        'actual_tasks_per_second': divide(totals['Total'], duration),
        'reached_tasks': reached, 'lost_tasks': lost,
        'loss_breakdown_json': json.dumps(loss_breakdown, sort_keys=True),
        'loss_breakdown_total': sum(loss_breakdown.values()),
        **{key.lower(): value for key, value in loss_breakdown.items()},
        'unfinished_tasks': totals['Total'] - finished,
        'finished_tasks_percent': divide(finished * 100, totals['Total']),
        'lost_generated_percent': divide(lost * 100, totals['Total']),
        'packet_loss_rate_percent': divide(lost * 100, finished),
        'average_delay_seconds': average_delay,
        'average_computation_delay_seconds': components['computation'],
        'average_queueing_delay_seconds': components['queueing'],
        'average_propagation_delay_seconds': components['propagation'],
        'average_transmission_delay_seconds': components['transmission'],
        'average_delay_component_sum_seconds': component_sum,
        'average_delay_accounting_gap_seconds': (
            average_delay - component_sum
            if average_delay is not None and component_sum is not None else None
        ),
        'average_hop_count': divide(
            totals['Total_hops_0'] + totals['Total_hops_1'], reached
        ),
        'average_computing_percent': divide(
            totals['Is_computing'] * 100,
            (TEST_BASE_SATELLITE_COUNT if satellite_count is None else satellite_count)
            * duration,
        ),
        'average_waiting_seconds': divide(totals['Computing_waiting_time'], reached),
        'average_ending_reward': divide(sum(rewards), len(rewards)),
        'workload_trace_count': parse_test_optional_number(content, 'Workload trace count'),
        'workload_trace_sha256': parse_test_optional_text(content, 'Workload trace SHA256'),
        'fault_trace_count': parse_test_optional_number(content, 'Fault trace count'),
        'fault_trace_sha256': parse_test_optional_text(content, 'Fault trace SHA256'),
        'raw_result_file': str(raw_path),
    }


def parse_test_optional_number(text, label):
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    match = re.search(rf'{re.escape(label)}: ({number}|None)', text)
    return None if not match or match.group(1) == 'None' else float(match.group(1))


def parse_test_optional_text(text, label):
    match = re.search(rf'{re.escape(label)}: ([^\r\n]+)', text)
    return match.group(1).strip() if match else None

CONFIG_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test' / 'config'
OUTPUT_ROOT = (
    PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test'
    / 'fault' / 'self_elimination'
)
MODEL_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'model' / 'ISACR_fault_self_elimination'
ALGORITHM = 'ISACR_fault_self_elimination'
ALL_CONSTELLATIONS = ('Walker', 'OneWeb', 'Starlink', 'Telesat')
CONSTELLATIONS = ALL_CONSTELLATIONS
CONFIG_ROOTS = {
    'Walker': CONFIG_ROOT / 'sat' / 'Walker',
    'OneWeb': CONFIG_ROOT / 'sat' / 'OneWeb',
    'Starlink': CONFIG_ROOT / 'sat' / 'Starlink',
    'Telesat': CONFIG_ROOT / 'sat' / 'Telesat',
}
COEFFICIENT_CONFIG_NAME = 'coefficient_experiments.json'
COEFFICIENT_CONFIG_PATH = CONFIG_ROOT / COEFFICIENT_CONFIG_NAME
COEFFICIENTS = tuple(
    float(value) for value in json.loads(
        COEFFICIENT_CONFIG_PATH.read_text(encoding='utf-8')
    )['coefficients']
)
TEST_SEED = 2025
PYTHON_HASH_SEED = str(TEST_SEED)
os.environ['PYTHONHASHSEED'] = PYTHON_HASH_SEED
EXPECTED_TEST_LOAD = {
    'target_load_tasks_per_second': 160,
    'poisson_rate_g': 14.57,
    'poisson_rate_s': 29.15,
}
TARGET_LOAD = EXPECTED_TEST_LOAD['target_load_tasks_per_second']
FAULT_PERCENT = 18
FAULT_PROBABILITY = 0.18
DURATION = 300
EXPECTED_STEP = 300
MAX_WORKERS = 20
TOTAL_MODELS = len(COEFFICIENTS) * len(SEEDS)
TOTAL_CASES = TOTAL_MODELS * len(CONSTELLATIONS)
LOCK_PATH = OUTPUT_ROOT / 'UNIFIED_TESTS_RUNNING.lock'
SIMULATION_SOURCE_HASHES = {
    name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
    for name in (
        'SatelliteNetworkSimulator_Computing.py',
        'SatelliteNetworkSimulator_Beta.py',
        'RL_environment_for_computing.py',
        'Base_Agents.py',
        'training_runner.py',
        'train_ISACR_self_elimination.py',
        'train_5fix_5seed.py',
        'test_5ix_5seed_5sat.py',
        'test_config.py',
    )
}

MEAN_FIELDS = (
    'actual_tasks_per_second',
    'generated_tasks',
    'reached_tasks',
    'lost_tasks',
    'unfinished_tasks',
    'finished_tasks_percent',
    'lost_generated_percent',
    'packet_loss_rate_percent',
    'average_delay_seconds',
    'average_computation_delay_seconds',
    'average_queueing_delay_seconds',
    'average_propagation_delay_seconds',
    'average_transmission_delay_seconds',
    'average_delay_component_sum_seconds',
    'average_delay_accounting_gap_seconds',
    'average_hop_count',
    'average_computing_percent',
    'average_waiting_seconds',
    'average_ending_reward',
)


ISACR_PLUS = False
BASE_OUTPUT_ROOT = OUTPUT_ROOT


def configure_isacr_plus(enabled):
    global ISACR_PLUS, OUTPUT_ROOT, LOCK_PATH
    if ISACR_PLUS == enabled:
        return
    ISACR_PLUS = enabled
    OUTPUT_ROOT = BASE_OUTPUT_ROOT.with_name(BASE_OUTPUT_ROOT.name + '_plus') if enabled else BASE_OUTPUT_ROOT
    LOCK_PATH = OUTPUT_ROOT / 'UNIFIED_TESTS_RUNNING.lock'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def coefficient_tag(value):
    return str(float(value)).replace('.', 'p')


def last_step(path):
    if not path.is_file():
        return None
    matches = re.findall(
        r'^====== step (\d+) ======$',
        path.read_text(encoding='utf-8', errors='replace'),
        re.MULTILINE,
    )
    return int(matches[-1]) if matches else None


def checkpoint_reward(path):
    match = re.search(r'_reward(-?\d+p\d+)', path.name)
    if not match:
        return float('-inf')
    return float(match.group(1).replace('p', '.'))


def count_satellites(tle_path):
    return sum(
        line.startswith('1 ')
        for line in tle_path.read_text(encoding='utf-8').splitlines()
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8',
    )
    os.replace(temporary, path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text('', encoding='utf-8-sig')
        os.replace(temporary, path)
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_trace(config, entry):
    root = config['root'].resolve()
    path = (root / entry['path']).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    if digest(path) != entry['sha256']:
        raise ValueError(f"{config['name']}: trace SHA256 mismatch: {path}")
    return path


def load_constellation_config(name):
    root = CONFIG_ROOTS[name]
    common = read_json(root / 'common.json')
    load_experiments = read_json(root / 'load_experiments.json')
    fault_experiments = read_json(root / 'fault_experiments.json')
    scenario_seed = int(common['general']['scenario_seed'])
    manifest = read_json(root / f'scenario_seed_{scenario_seed}' / 'manifest.json')

    load_cases = [
        item for item in load_experiments['cases']
        if int(item['target_load_tasks_per_second']) == TARGET_LOAD
    ]
    fault_cases = [
        item for item in fault_experiments['cases']
        if abs(float(item['edge_p']) - FAULT_PROBABILITY) < 1e-12
    ]
    if len(load_cases) != 1 or len(fault_cases) != 1:
        raise ValueError(f'{name}: expected one load-160 and one fault-18% case')
    if scenario_seed != TEST_SEED:
        raise ValueError(f'{name}: unexpected scenario seed {scenario_seed}')
    if int(common['general']['duration']) != DURATION:
        raise ValueError(f'{name}: unexpected duration')
    if (
        common['general'].get('phase') != 'test'
        or common['general'].get('epsilon') != 0.0
        or common['general'].get('random_seed') != TEST_SEED
        or str(common['python_hash_seed']) != PYTHON_HASH_SEED
        or DURATION // int(common['general']['time_stride']) != EXPECTED_STEP
        or int(manifest['scenario_seed']) != TEST_SEED
        or int(manifest['duration']) != DURATION
    ):
        raise ValueError(f'{name}: test settings or trace scenario are not aligned')
    if common['stream_contract'].get('mode') != 'fixed_trace_files':
        raise ValueError(f'{name}: test is not using fixed trace files')
    if list(common['general'].get('reward_factors', ())) != [1, 0.05, 1, 0.1, 0.25]:
        raise ValueError(f'{name}: reward factors are not aligned')
    if common['environment'].get('require_computed_before_downlink') is not False:
        raise ValueError(f'{name}: direct-downlink target rule is disabled')
    if common['stream_contract'].get('destination_policy') != (
        'fixed_session_destination_direct_downlink_at_target'
    ):
        raise ValueError(f'{name}: destination policy is not fixed-target direct downlink')
    if abs(float(load_experiments.get('edge_p', -1)) - FAULT_PROBABILITY) > 1e-12:
        raise ValueError(f'{name}: load tests must use 18% fault probability')
    if abs(float(load_experiments.get('edge_q', -1)) - (1.0 - FAULT_PROBABILITY)) > 1e-12:
        raise ValueError(f'{name}: load tests have inconsistent edge_q')
    fault_workload = fault_experiments.get('workload', {})
    if (
        int(fault_workload.get('target_load_tasks_per_second', -1))
        != EXPECTED_TEST_LOAD['target_load_tasks_per_second']
        or float(fault_workload.get('poisson_rate_g', -1))
        != EXPECTED_TEST_LOAD['poisson_rate_g']
        or float(fault_workload.get('poisson_rate_s', -1))
        != EXPECTED_TEST_LOAD['poisson_rate_s']
    ):
        raise ValueError(f'{name}: fault tests must use load 160 Poisson rates')
    if any(load_cases[0].get(key) != value for key, value in EXPECTED_TEST_LOAD.items()):
        raise ValueError(f'{name}: load-160 settings are not aligned')
    if float(fault_cases[0]['edge_q']) != 1.0 - FAULT_PROBABILITY:
        raise ValueError(f'{name}: fault case has inconsistent edge_q')
    for path_key, hash_key in (
        ('tle_filepath', 'tle_sha256'),
        ('computing_ability_filepath', 'computing_ability_sha256'),
    ):
        resource_path = (PROJECT_ROOT / common['environment']['tle_filepath']).resolve()
        if path_key == 'computing_ability_filepath':
            resource_path = resource_path.with_name(f'{resource_path.stem}_computing_ability.csv')
        if (
            resource_path != (PROJECT_ROOT / manifest[path_key]).resolve()
            or digest(resource_path) != manifest[hash_key]
        ):
            raise ValueError(f'{name}: constellation resource mismatch: {path_key}')
    if common['environment']['computing_ability'] is not None:
        raise ValueError(f'{name}: per-satellite computing ability is required')

    config_names = (
        'common.json',
        'load_experiments.json',
        'fault_experiments.json',
        f'scenario_seed_{scenario_seed}/manifest.json',
    )
    hashes = {item: digest(root / item) for item in config_names}
    workload = manifest['workloads']['load_160']
    fault_trace = manifest['faults']['fault_p0p18']
    data = {
        'name': name,
        'root': root,
        'common': common,
        'load': load_cases[0],
        'fault': fault_cases[0],
        'manifest': manifest,
        'hashes': hashes,
        'contract_hash': hashlib.sha256(
            json.dumps({
                'configuration': hashes,
                'simulation_source': SIMULATION_SOURCE_HASHES,
                'runtime_acceleration': runtime_acceleration_contract(),
            }, sort_keys=True).encode('ascii')
        ).hexdigest(),
        'workload': workload,
        'fault_trace': fault_trace,
    }
    data['workload_path'] = resolve_trace(data, workload)
    data['fault_trace_path'] = resolve_trace(data, fault_trace)
    tle_path = (PROJECT_ROOT / common['environment']['tle_filepath']).resolve()
    if not tle_path.is_file():
        raise FileNotFoundError(tle_path)
    data['tle_path'] = tle_path
    data['satellite_count'] = count_satellites(tle_path)
    if data['satellite_count'] != int(manifest['satellite_count']):
        raise ValueError(f'{name}: satellite count does not match trace manifest')
    return data


CONFIGS = {name: load_constellation_config(name) for name in CONSTELLATIONS}


def model_directory(seed, coefficient):
    return MODEL_ROOT / f'seed{seed}' / f'coefficient_{coefficient_tag(coefficient)}'


def select_model(seed, coefficient):
    directory = model_directory(seed, coefficient)
    result_path = directory / 'process_result.json'
    config_path = directory / 'training_config.json'
    if not result_path.is_file() or not config_path.is_file():
        raise RuntimeError(f'{directory}: training result or configuration not found')
    result = read_json(result_path)
    config = read_json(config_path)
    from train_ISACR_self_elimination import CONFIG
    if (
        result.get('exit_code') != 0
        or config.get('agent', {}).get('neighbors_dim') != CONFIG['agent']['neighbors_dim']
        or result.get('final_step') != EXPECTED_TRAINING_STEP
        or config.get('general', {}).get('random_seed') != seed
        or config.get('general', {}).get('scenario_seed') != EXPECTED_TRAINING_SCENARIO_SEED
        or config.get('general', {}).get('duration') != EXPECTED_TRAINING_STEP
        or config.get('general', {}).get('training_reward_save_start_step') != EXPECTED_TRAINING_REWARD_SAVE_START_STEP
        or config.get('general', {}).get('model_saving_enabled') is not True
        or config.get('agent', {}).get('update_cycle') != UPDATE_CYCLE
        or config.get('environment', {}).get('fault_fill_multiplier') != coefficient
        or config.get('environment', {}).get('edge_p') != EXPECTED_TRAINING_EDGE_P
        or config.get('environment', {}).get('edge_q') != EXPECTED_TRAINING_EDGE_Q
        or any(
            config.get('environment', {}).get(key) != value
            for key, value in EXPECTED_TRAINING_LOAD.items()
        )
    ):
        raise RuntimeError(
            f'{directory}: training is incomplete or failed: {result}'
        )
    checkpoint = result.get('reward_checkpoint')
    if not checkpoint or not Path(checkpoint).is_file():
        raise RuntimeError(f'{directory}: reward-best checkpoint not found')
    training_log = result.get('training_log')
    if not training_log or last_step(Path(training_log)) != EXPECTED_TRAINING_STEP:
        raise RuntimeError(
            f'{directory}: no complete training log at step '
            f'{EXPECTED_TRAINING_STEP}'
        )
    checkpoint = Path(checkpoint)
    checkpoint_config = load_checkpoint_training_config(checkpoint)
    if any(checkpoint_config[section] != config[section]
           for section in ('general', 'agent', 'environment')):
        raise RuntimeError(f'{directory}: checkpoint metadata differs from training config')
    build_test_config_base(EXPECTED_TRAINING_LOAD, checkpoint, coefficient)
    return checkpoint


def case_directory(constellation, coefficient, seed):
    return (
        OUTPUT_ROOT / constellation
        / f'seed{seed}'
        / f'coefficient_{coefficient_tag(coefficient)}'
    )


def build_case_config(constellation, checkpoint, coefficient, raw_path):
    data = CONFIGS[constellation]
    config = build_test_config_base(data['load'], checkpoint, coefficient)
    for key, value in data['common']['general'].items():
        if key in {'phase', 'random_seed', 'scenario_seed', 'duration', 'epsilon',
                   'epsilon_decay', 'min_epsilon', 'print_cycle', 'show_test_summary'}:
            continue
        if config['general'].get(key) != value:
            raise ValueError(f'Training/constellation setting mismatch: general.{key}')
    for key, value in data['common']['environment'].items():
        if key in {'tle_filepath', 'computing_ability', 'fault_fill_multiplier',
                   'visualize', 'print_info', 'save_log', 'show_detail'}:
            continue
        if config['environment'].get(key) != value:
            raise ValueError(f'Training/constellation setting mismatch: environment.{key}')
    config['general'].update(data['common']['general'])
    config['general'].update(
        random_seed=TEST_SEED,
        scenario_seed=TEST_SEED,
        print_cycle=1,
        show_test_summary=True,
    )
    config['agent']['model_path'] = str(checkpoint)
    config['agent']['update_cycle'] = UPDATE_CYCLE
    config['environment'].update(data['common']['environment'])
    config['environment'].update(data['load'])
    config['environment'].update(
        edge_p=FAULT_PROBABILITY,
        edge_q=1.0 - FAULT_PROBABILITY,
        node_p=0.0,
        node_q=0.0,
        markov_mode=True,
        fault_compute_double_mean=True,
        fault_fill_multiplier=coefficient,
        require_computed_before_downlink=False,
        visualize=False,
        save_log=False,
        save_training_data=str(raw_path),
        workload_trace_path=str(data['workload_path']),
        workload_trace_sha256=data['workload']['sha256'],
        workload_trace_task_count=int(data['workload']['task_count']),
        fault_trace_path=str(data['fault_trace_path']),
        fault_trace_sha256=data['fault_trace']['sha256'],
        fault_trace_event_count=int(data['fault_trace']['event_count']),
    )
    config['test_contract'] = {
        'runtime_acceleration': runtime_acceleration_contract(),
        'constellation': constellation,
        'case_id': 'load_160_fault_p0p18',
        'contract_sha256': data['contract_hash'],
        'configuration_sha256': data['hashes'],
        'simulation_source_sha256': SIMULATION_SOURCE_HASHES,
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': digest(checkpoint),
        'python_hash_seed': str(data['common']['python_hash_seed']),
        'stream_contract': data['common']['stream_contract'],
        'workload_trace': data['workload'],
        'fault_trace': data['fault_trace'],
    }
    config['coefficient_test'] = {
        'config_sha256': digest(COEFFICIENT_CONFIG_PATH),
        'coefficient': coefficient,
        'fault_percent': FAULT_PERCENT,
        'target_load_tasks_per_second': TARGET_LOAD,
        'training_seed': None,
    }
    config['test_contract']['isacr_plus'] = ISACR_PLUS
    if ISACR_PLUS:
        config['general']['select_mode'] = 3
    return config


def compatible_contract_hashes(data):
    # Only the coefficient runner/plot scripts changed; simulation sources stay exact.
    hashes = {data['contract_hash']}
    for training_hash, testing_hash in (
        ('22a5e9fe3e182a8ed284c61fb6ef974a4f346fc96ea1c4223f22b69306ce659e',
         '2e0d64ab70e9b1f2e85dc42cfa183b275fd4ff1d4b65ae1b0c698b2ae9d23228'),
        ('29c1b02641fd15787a7d73820b38c9cec2efb539b5e6ff77ac12910bff1c887b',
         'e6d3c298440cd3bc88cf9236d47dfe8619c906c092a90d4e52347c91ad109f25'),
    ):
        previous_sources = dict(SIMULATION_SOURCE_HASHES)
        previous_sources.pop('train_5fix_5seed.py')
        previous_sources.pop('test_5ix_5seed_5sat.py')
        previous_sources.update({'train_5fix_5seed.py': training_hash,
                                 'test_5ix_5seed_5sat.py': testing_hash})
        hashes.add(hashlib.sha256(json.dumps({
            'configuration': data['hashes'], 'simulation_source': previous_sources,
            'runtime_acceleration': runtime_acceleration_contract(),
        }, sort_keys=True).encode('ascii')).hexdigest())
    return hashes


def coefficient_config_hashes(coefficient):
    hashes = {digest(COEFFICIENT_CONFIG_PATH)}
    for previous in COEFFICIENT_CONFIG_PATH.parent.glob('coefficient_experiments_previous*.json'):
        current_config = read_json(COEFFICIENT_CONFIG_PATH)
        previous_config = read_json(previous)
        previous_coefficients = previous_config.pop('coefficients')
        current_config.pop('coefficients')
        if previous_config == current_config and coefficient in previous_coefficients:
            hashes.add(digest(previous))
    return hashes


def case_complete(constellation, coefficient, seed):
    directory = case_directory(constellation, coefficient, seed)
    raw_path = directory / 'result.txt'
    config_path = directory / 'config.json'
    summary_path = directory / 'summary.json'
    process_result_path = directory / 'process_result.json'
    if (
        last_step(raw_path) != EXPECTED_STEP
        or not config_path.is_file()
        or not summary_path.is_file()
        or not process_result_path.is_file()
    ):
        return False
    try:
        config = read_json(config_path)
        summary = read_json(summary_path)
        process_result = read_json(process_result_path)
        coefficient_test = config['coefficient_test']
        contract = config['test_contract']
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    try:
        checkpoint = select_model(seed, coefficient)
        checkpoint_sha256 = digest(checkpoint)
    except (OSError, RuntimeError):
        return False
    data = CONFIGS[constellation]
    return (
        contract.get('isacr_plus', False) == ISACR_PLUS
        and config['general'].get('select_mode') == (3 if ISACR_PLUS else 1)
        and process_result.get('exit_code') == 0
        and process_result.get('model_sha256') == checkpoint_sha256
        and contract.get('checkpoint_sha256') == checkpoint_sha256
        and contract.get('checkpoint') == str(checkpoint.resolve())
        and config['agent'].get('update_cycle') == UPDATE_CYCLE
        and coefficient_test.get('config_sha256') in coefficient_config_hashes(coefficient)
        and float(coefficient_test.get('coefficient', -1)) == coefficient
        and int(coefficient_test.get('training_seed', -1)) == seed
        and coefficient_test.get('target_load_tasks_per_second') == TARGET_LOAD
        and coefficient_test.get('fault_percent') == FAULT_PERCENT
        and contract.get('constellation') == constellation
        and contract.get('contract_sha256') in compatible_contract_hashes(data)
        and contract.get('workload_trace', {}).get('sha256') == data['workload']['sha256']
        and contract.get('fault_trace', {}).get('sha256') == data['fault_trace']['sha256']
        and summary.get('constellation') == constellation
        and int(summary.get('training_seed', -1)) == seed
        and float(summary.get('fault_fill_multiplier', -1)) == coefficient
        and float(summary.get('edge_p', -1)) == FAULT_PROBABILITY
        and summary.get('target_load_tasks_per_second') == TARGET_LOAD
        and summary.get('model_sha256') == checkpoint_sha256
        and summary.get('test_contract_sha256') in compatible_contract_hashes(data)
        and summary.get('workload_trace_sha256') == data['workload']['sha256']
        and summary.get('fault_trace_sha256') == data['fault_trace']['sha256']
    )


def run_case(job):
    constellation, coefficient, seed, checkpoint_text, force = job
    checkpoint = Path(checkpoint_text)
    directory = case_directory(constellation, coefficient, seed)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / 'result.txt'
    running_path = directory / 'result.running.txt'
    config_path = directory / 'config.json'
    summary_path = directory / 'summary.json'
    console_path = directory / 'console.txt'
    process_result_path = directory / 'process_result.json'

    if not force and case_complete(constellation, coefficient, seed):
        return read_json(summary_path)

    archive_test_outputs((raw_path, running_path, config_path, summary_path,
                          console_path, process_result_path))
    config = build_case_config(constellation, checkpoint, coefficient, running_path)
    config['coefficient_test']['training_seed'] = seed
    prepare_test_acceleration(config)
    write_json(config_path, config)
    with console_path.open('w', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            with destination_selection(ablation=False) if ISACR_PLUS else nullcontext():
                run_training(config)
    if last_step(running_path) != EXPECTED_STEP:
        raise RuntimeError(
            f'Incomplete result: {running_path}; expected {EXPECTED_STEP}, '
            f'got {last_step(running_path)}'
        )
    running_path.replace(raw_path)
    config['environment']['save_training_data'] = str(raw_path)
    write_json(config_path, config)
    data = CONFIGS[constellation]
    row = parse_test_result(
        raw_path,
        ALGORITHM,
        data['load'],
        DURATION,
        satellite_count=data['satellite_count'],
    )
    row.update(
        isacr_plus=ISACR_PLUS,
        constellation=constellation,
        training_seed=seed,
        scenario_seed=TEST_SEED,
        fault_percent=FAULT_PERCENT,
        fault_fill_multiplier=coefficient,
        edge_p=FAULT_PROBABILITY,
        edge_q=1.0 - FAULT_PROBABILITY,
        test_type='load_160_fault_p0p18',
        model_path=str(checkpoint),
        model_sha256=config['test_contract']['checkpoint_sha256'],
        test_contract_sha256=data['contract_hash'],
        expected_workload_trace_sha256=data['workload']['sha256'],
        expected_fault_trace_sha256=data['fault_trace']['sha256'],
    )
    write_json(summary_path, row)
    write_json(
        process_result_path,
        {
            'exit_code': 0,
            'model_sha256': config['test_contract']['checkpoint_sha256'],
            'finished_at': datetime.now().astimezone().isoformat(),
        },
    )
    return row


def find_ready_models():
    ready = {}
    errors = {}
    for coefficient in COEFFICIENTS:
        for seed in SEEDS:
            try:
                ready[(coefficient, seed)] = select_model(seed, coefficient)
            except Exception as error:
                errors[f'coefficient_{coefficient_tag(coefficient)}/seed{seed}'] = str(error)
    return ready, errors


def missing_jobs(ready, force=False):
    return [
        (constellation, coefficient, seed, str(checkpoint), force)
        for constellation in CONSTELLATIONS
        for (coefficient, seed), checkpoint in ready.items()
        if force or not case_complete(constellation, coefficient, seed)
    ]


def collect_rows(ready):
    rows = []
    for (coefficient, seed) in ready:
        for constellation in CONSTELLATIONS:
            if case_complete(constellation, coefficient, seed):
                rows.append(read_json(
                    case_directory(constellation, coefficient, seed) / 'summary.json'
                ))
    return sorted(
        rows,
        key=lambda row: (
            row['constellation'],
            float(row['fault_fill_multiplier']),
            int(row['training_seed']),
        ),
    )


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['constellation'], float(row['fault_fill_multiplier']))].append(row)
    output = []
    for (constellation, coefficient), items in sorted(groups.items()):
        result = {
            'constellation': constellation,
            'fault_fill_multiplier': coefficient,
            'fault_percent': FAULT_PERCENT,
            'target_load_tasks_per_second': TARGET_LOAD,
            'seed_count': len({int(item['training_seed']) for item in items}),
        }
        for field in MEAN_FIELDS:
            values = [numeric(item.get(field)) for item in items]
            values = [value for value in values if value is not None]
            if values:
                result[field] = sum(values) / len(values)
        output.append(result)
    return output


def write_plots(rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['constellation'], float(row['fault_fill_multiplier']))].append(row)
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=200)
    metrics = (
        ('packet_loss_rate_percent', 'Packet loss rate (%)'),
        ('average_delay_seconds', 'Average delay (s)'),
        ('average_hop_count', 'Average hop count'),
        ('average_ending_reward', 'Average reward'),
    )
    for axis, (field, ylabel) in zip(axes, metrics):
        for coefficient in COEFFICIENTS:
            values = []
            for constellation in CONSTELLATIONS:
                items = grouped[(constellation, coefficient)]
                values.append(
                    sum(float(item[field]) for item in items) / len(items)
                    if items else None
                )
            axis.plot(
                CONSTELLATIONS,
                values,
                marker='o',
                linewidth=1.6,
                label=f'coefficient {coefficient:g}',
            )
        axis.set_xlabel('Constellation')
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle='--', linewidth=0.6, alpha=0.55)
        axis.set_axisbelow(True)
    figure.suptitle('ISACR fault self-elimination: load 160, fault rate 18%')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.93))
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    figure.savefig(OUTPUT_ROOT / 'four_constellation_comparison.png', bbox_inches='tight')
    figure.savefig(OUTPUT_ROOT / 'four_constellation_comparison.pdf', bbox_inches='tight')
    plt.close(figure)


def process_is_running(process_id):
    try:
        os.kill(process_id, 0)
    except (OSError, ValueError):
        return False
    return True


def acquire_lock():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.is_file():
        try:
            process_id = int(LOCK_PATH.read_text(encoding='ascii').strip())
        except (OSError, ValueError):
            process_id = None
        if process_id is not None and process_is_running(process_id):
            return False
        LOCK_PATH.unlink(missing_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w', encoding='ascii') as stream:
        stream.write(str(os.getpid()))
    return True


def save_status(ready, rows, model_errors, run_errors):
    missing = [
        f'{constellation}/seed{seed}/coefficient_{coefficient_tag(coefficient)}'
        for coefficient in COEFFICIENTS
        for seed in SEEDS
        for constellation in CONSTELLATIONS
        if not case_complete(constellation, coefficient, seed)
    ]
    complete = (
        len(ready) == TOTAL_MODELS
        and len(rows) == TOTAL_CASES
        and not model_errors
        and not run_errors
        and not missing
    )
    status = {
        'status': 'complete' if complete else 'partial',
        'algorithm': ALGORITHM,
        'update_cycle': UPDATE_CYCLE,
        'constellations': list(CONSTELLATIONS),
        'training_seeds': list(SEEDS),
        'coefficients': list(COEFFICIENTS),
        'target_load_tasks_per_second': TARGET_LOAD,
        'fault_percent': FAULT_PERCENT,
        'fault_probability': FAULT_PROBABILITY,
        'duration_seconds': DURATION,
        'test_scenario_seed': TEST_SEED,
        'python_hash_seed': PYTHON_HASH_SEED,
        'simulation_source_sha256': SIMULATION_SOURCE_HASHES,
        'expected_models': TOTAL_MODELS,
        'ready_models': len(ready),
        'expected_cases': TOTAL_CASES,
        'completed_cases': len(rows),
        'missing_cases': missing,
        'model_errors': model_errors,
        'run_errors': run_errors,
        'max_workers': MAX_WORKERS,
        'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
    }
    write_json(OUTPUT_ROOT / 'STATUS.json', status)
    marker = OUTPUT_ROOT / 'ALL_COMPLETE'
    if complete:
        marker.write_text(f'{TOTAL_CASES}/{TOTAL_CASES}\n', encoding='utf-8')
    else:
        marker.unlink(missing_ok=True)
    return status


def main():
    global CONSTELLATIONS, TOTAL_CASES, COEFFICIENTS, TOTAL_MODELS
    if COEFFICIENTS != FILL_MULTIPLIERS:
        raise ValueError(
            f'coefficient_experiments.json must match training coefficients {FILL_MULTIPLIERS}, '
            f'got {COEFFICIENTS}'
        )

    parser = argparse.ArgumentParser(description=__doc__)
    add_acceleration_argument(parser)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--status-only', action='store_true')
    parser.add_argument(
        '--rerun', action='store_true',
        help='Retest all 25 five-coefficient, five-seed models at load 160/fault 18%% '
             'on selected constellations (100 cases by default).',
    )
    constellation_group = parser.add_mutually_exclusive_group()
    constellation_group.add_argument('--constellation', choices=ALL_CONSTELLATIONS)
    constellation_group.add_argument('--constellations', nargs='+', choices=ALL_CONSTELLATIONS,
                                     help='Share the worker pool across selected constellations.')
    parser.add_argument('--coefficients', nargs='+', type=float, choices=FILL_MULTIPLIERS,
                        help='Test only the selected trained coefficients.')
    parser.add_argument('--isacr-plus', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable joint destination/action selection using existing weights.')
    args = parser.parse_args()
    configure_acceleration(args.acceleration == 'on')
    configure_isacr_plus(args.isacr_plus)
    if args.coefficients is not None:
        COEFFICIENTS = tuple(dict.fromkeys(args.coefficients))
        TOTAL_MODELS = len(COEFFICIENTS) * len(SEEDS)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f'--workers must be between 1 and {MAX_WORKERS}')
    CONSTELLATIONS = (
        tuple(dict.fromkeys(args.constellations)) if args.constellations else
        (args.constellation,)
        if args.constellation is not None
        else ALL_CONSTELLATIONS
    )
    TOTAL_CASES = TOTAL_MODELS * len(CONSTELLATIONS)
    if not acquire_lock():
        print(f'STATUS=busy: {LOCK_PATH}', flush=True)
        return

    model_errors = {}
    run_errors = {}
    try:
        ready, model_errors = find_ready_models()
        jobs = [] if args.status_only else missing_jobs(ready, args.rerun)
        if jobs:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                     initializer=configure_isacr_plus,
                                     initargs=(args.isacr_plus,)) as executor:
                futures = {executor.submit(run_case, job): job for job in jobs}
                for index, future in enumerate(as_completed(futures), start=1):
                    job = futures[future]
                    try:
                        future.result()
                        print(
                            f'Completed {index}/{len(jobs)}: '
                            f'{job[0]}/seed{job[2]}/coefficient_{coefficient_tag(job[1])}',
                            flush=True,
                        )
                    except Exception as error:
                        key = f'{job[0]}/seed{job[2]}/coefficient_{coefficient_tag(job[1])}'
                        run_errors[key] = str(error)
                        print(f'ERROR {key}: {error}', flush=True)
        rows = collect_rows(ready)
        write_rows(OUTPUT_ROOT / 'all_results.csv', rows)
        write_rows(OUTPUT_ROOT / 'mean_by_constellation_coefficient.csv', mean_rows(rows))
        write_plots(rows)
        status = save_status(ready, rows, model_errors, run_errors)
        print(f'READY_MODELS={status["ready_models"]}/{TOTAL_MODELS}', flush=True)
        print(f'COMPLETE_CASES={status["completed_cases"]}/{TOTAL_CASES}', flush=True)
        print(f'ALL_COMPLETE={str(status["status"] == "complete").lower()}', flush=True)
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
