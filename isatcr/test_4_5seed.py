"""Run resumable six-method, five-seed load and fault experiments."""

import argparse
from test_config import (
    add_acceleration_argument, configure_acceleration, prepare_test_acceleration, run_training,
    destination_selection,
)
import ast
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from training_runner import find_latest_checkpoint, load_checkpoint_training_config, runtime_acceleration_contract, archive_test_outputs
from train_4_5seed import (
    build_training_config,
    validate_training_config,
    TRAINING_DURATION as EXPECTED_TRAINING_STEP,
    TRAINING_REWARD_SAVE_START_STEP as EXPECTED_TRAINING_REWARD_SAVE_START_STEP,
    TRAINING_SCENARIO_SEED as EXPECTED_TRAINING_SCENARIO_SEED,
    TRAINING_LOAD as EXPECTED_TRAINING_LOAD,
    TRAINING_EDGE_P as EXPECTED_TRAINING_EDGE_P,
    TRAINING_EDGE_Q as EXPECTED_TRAINING_EDGE_Q,
    TRAINING_FAULT_FILL_MULTIPLIER,
)


PROJECT_ROOT = Path(__file__).resolve().parent
COMPUTING_ABILITY_FILE = (
    PROJECT_ROOT / 'Satellite_Data' / 'wallker_tles_computing_ability.csv'
)

# Local test configuration and result parser.  This script is standalone.
TEST_BASE_SATELLITE_COUNT = sum(
    line.startswith('1 ')
    for line in (PROJECT_ROOT / 'Satellite_Data' / 'wallker_tles.txt')
    .read_text(encoding='utf-8').splitlines()
)
TEST_BASE_PACKET_FREQUENCY = 1
TEST_BASE_TRAINING_SEED = 0

def build_test_config_base(algorithm, load_setting, checkpoint):
    model_algorithm = MODEL_ALGORITHMS.get(algorithm, algorithm)
    if model_algorithm == 'ISACR_self_elimination':
        from test_5ix_5seed_5sat import load_checkpoint_training_config as load_isacr_config
        config = load_isacr_config(checkpoint)
    else:
        config = load_checkpoint_training_config(checkpoint)
    if (model_algorithm == 'ISACR_self_elimination'
            and config['general']['algorithm_name'] == 'ISACR_fault_self_elimination'):
        config['general']['algorithm_name'] = model_algorithm
    if config['general']['algorithm_name'] != model_algorithm:
        raise ValueError(f'Checkpoint algorithm mismatch: {checkpoint}')
    training_config = build_training_config(
        model_algorithm, config['general']['random_seed'],
    )
    # Keep the trained decision logic; only the evaluation scenario may change.
    evaluation_fields = {
        'general': {'phase', 'random_seed', 'scenario_seed', 'duration', 'rounds',
                    'epsilon', 'epsilon_decay', 'min_epsilon', 'print_cycle',
                    'show_test_summary', 'model_saving_enabled',
                    'training_reward_save_start_step', 'metric_checkpoint_names'},
        'agent': {'model_path', 'initial_model_path'},
        'environment': {'id', 'target_load_tasks_per_second', 'poisson_rate_s',
                        'poisson_rate_g', 'edge_p', 'edge_q', 'computing_ability',
                        'visualize', 'print_info', 'save_log', 'show_detail',
                        'save_training_data'},
    }
    for section, excluded in evaluation_fields.items():
        for key in training_config[section].keys() | config[section].keys():
            if key in excluded:
                continue
            expected = training_config[section].get(key)
            if (key not in training_config[section] or key not in config[section]
                    or json.dumps(config[section][key]) != json.dumps(expected)):
                raise ValueError(
                    f'Checkpoint/current training setting mismatch: '
                    f'{model_algorithm}.{section}.{key}; use a matching trained model'
                )
    for key in ('reward_factors', 'select_mode', 'begin_time', 'time_stride'):
        if config['general'].get(key) != COMMON_TEST['general'].get(key):
            raise ValueError(f'Training/common test setting mismatch: general.{key}')
    for key, value in COMMON_TEST['environment'].items():
        if key in {'computing_ability', 'visualize', 'print_info', 'save_log', 'show_detail'}:
            continue
        if config['environment'].get(key) != value:
            raise ValueError(f'Training/common test setting mismatch: environment.{key}')
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

DEFAULT_MODEL_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'model'
MODEL_ROOT = Path(os.environ.get('SATELLITE_MODEL_ROOT', DEFAULT_MODEL_ROOT))
BASE_TEST_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test'
TEST_ROOT = Path(os.environ.get('SATELLITE_TEST_ROOT', BASE_TEST_ROOT))
CONFIG_ROOT = Path(
    os.environ.get('SATELLITE_TEST_CONFIG_ROOT', BASE_TEST_ROOT / 'config')
)
TRAINING_STATUS_ROOT = Path(os.environ.get(
    'SATELLITE_TRAINING_STATUS_ROOT',
    DEFAULT_MODEL_ROOT / 'four_5seed_training_status',
))
LOCK_PATH = TEST_ROOT / 'FOUR_ALGORITHM_SEED_TESTS.running'
TRAINING_LOCKS = (
    DEFAULT_MODEL_ROOT / 'seed0_4_training_status' / 'TRAINING_RUNNING.lock',
    DEFAULT_MODEL_ROOT / 'four_5seed_training_status' / 'TRAINING_RUNNING.lock',
)
MODEL_MANIFEST_PATH = TEST_ROOT / 'four_algorithm_5seed_model_manifest.json'


def read_fixed_config(name):
    path = CONFIG_ROOT / name
    return json.loads(path.read_text(encoding='utf-8'))


def validate_per_satellite_computing_abilities():
    if not COMPUTING_ABILITY_FILE.is_file():
        raise FileNotFoundError(COMPUTING_ABILITY_FILE)
    with COMPUTING_ABILITY_FILE.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    required = {'satellite_name', 'computing_ability_gflops'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f'{COMPUTING_ABILITY_FILE} must contain {sorted(required)}'
        )
    if len(rows) != TEST_BASE_SATELLITE_COUNT:
        raise ValueError(
            'Per-satellite computing ability count does not match the TLE: '
            f'{len(rows)} != {TEST_BASE_SATELLITE_COUNT}'
        )
    abilities = [float(row['computing_ability_gflops']) for row in rows]
    if any(value <= 0 for value in abilities):
        raise ValueError('Per-satellite computing abilities must be positive')
    if len(set(abilities)) == 1:
        raise ValueError('Uniform computing ability is not allowed in four-algorithm tests')


def config_sha256(name):
    return hashlib.sha256((CONFIG_ROOT / name).read_bytes()).hexdigest()


COMMON_TEST = read_fixed_config('common.json')
COMMON_TEST['environment']['fault_fill_multiplier'] = TRAINING_FAULT_FILL_MULTIPLIER
LOAD_TESTS = read_fixed_config('load_experiments.json')
FAULT_TESTS = read_fixed_config('fault_experiments.json')
# Fixed conditions for this entry point; preserve other tests' shared configs.
LOAD_TESTS.update(edge_p=0.18, edge_q=0.82)
FAULT_TESTS['workload'] = {
    field: next(case for case in LOAD_TESTS['cases'] if case['id'] == 'load_160')[field]
    for field in ('target_load_tasks_per_second', 'poisson_rate_g', 'poisson_rate_s')
}
TEST_SEED = int(COMMON_TEST['general']['scenario_seed'])
RANDOM_SEED = int(COMMON_TEST['general']['random_seed'])
SCENARIO_MANIFEST_NAME = f'scenario_seed_{TEST_SEED}/manifest.json'
SCENARIO_MANIFEST = read_fixed_config(SCENARIO_MANIFEST_NAME)
CONFIG_HASHES = {
    name: config_sha256(name)
    for name in (
        'common.json',
        'load_experiments.json',
        'fault_experiments.json',
        SCENARIO_MANIFEST_NAME,
    )
}

ALGORITHMS = ('D3QN', 'DDQN', 'PPO', 'ISACR_self_elimination', 'D3QN+')
MODEL_ALGORITHMS = {
    'D3QN+': 'D3QN',
    'ISACR_self_elimination+': 'ISACR_self_elimination',
}
DESTINATION_SELECTION_ALGORITHMS = ('D3QN+', 'ISACR_self_elimination+')
DEFAULT_ALGORITHMS = ('ISACR_self_elimination', 'PPO', 'DDQN', 'D3QN')
TEST_ALGORITHMS = ALGORITHMS + ('ISACR_self_elimination+',)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def configured_training_seeds():
    raw = os.environ.get('SATELLITE_TRAINING_SEEDS')
    if raw is None:
        return DEFAULT_SEEDS
    try:
        seeds = tuple(int(item.strip()) for item in raw.split(',') if item.strip())
    except ValueError as error:
        raise ValueError(
            'SATELLITE_TRAINING_SEEDS must be a comma-separated integer list'
        ) from error
    if (
        not seeds
        or any(seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(
            'SATELLITE_TRAINING_SEEDS must contain unique non-negative integers'
        )
    return seeds


SEEDS = configured_training_seeds()
MAX_WORKERS = int(os.environ.get('SATELLITE_TEST_MAX_WORKERS', '10'))
DURATION = int(COMMON_TEST['general']['duration'])
EXPECTED_TEST_STEP = DURATION // int(COMMON_TEST['general']['time_stride'])
FAULT_FILL_MULTIPLIER = float(COMMON_TEST['environment']['fault_fill_multiplier'])
PYTHON_HASH_SEED = str(COMMON_TEST['python_hash_seed'])
os.environ['PYTHONHASHSEED'] = PYTHON_HASH_SEED

LOAD_SETTINGS = tuple(LOAD_TESTS['cases'])
LOAD_TEST_SEQUENCE = tuple(sorted(
    (int(item['target_load_tasks_per_second']) for item in LOAD_SETTINGS), reverse=True,
))
LOAD_SETTINGS_BY_POINT = {
    int(item['target_load_tasks_per_second']): item
    for item in LOAD_SETTINGS
}
LOAD_FAULT_PROBABILITY = float(LOAD_TESTS['edge_p'])
FAULT_LOAD_SETTING = dict(FAULT_TESTS['workload'])
FAULT_CASES = tuple(FAULT_TESTS['cases'])
FAULT_PROBABILITIES = tuple(float(item['edge_p']) for item in FAULT_CASES)
FAULT_TEST_SEQUENCE = tuple(sorted(FAULT_PROBABILITIES, reverse=True))
FAULT_CASE_BY_PROBABILITY = {
    float(item['edge_p']): item for item in FAULT_CASES
}
LOAD_FAULT_CASE = FAULT_CASE_BY_PROBABILITY[LOAD_FAULT_PROBABILITY]
FAULT_WORKLOAD_CASE = dict(
    FAULT_LOAD_SETTING,
    id=f"load_{int(FAULT_LOAD_SETTING['target_load_tasks_per_second'])}",
)
SIMULATION_SOURCE_HASHES = {
    name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
    for name in (
        'SatelliteNetworkSimulator_Computing.py',
        'SatelliteNetworkSimulator_Beta.py',
        'RL_environment_for_computing.py',
        'Base_Agents.py',
        'training_runner.py',
        'test_4_5seed.py',
        'test_config.py',
        'train_D3QN.py',
        'train_DDQN.py',
        'train_PPO.py',
        'train_ISACR_self_elimination.py',
    )
}
TEST_CONTRACT_HASH = hashlib.sha256(
    json.dumps({
        'configuration': CONFIG_HASHES,
        'simulation_source': SIMULATION_SOURCE_HASHES,
        'runtime_acceleration': runtime_acceleration_contract(),
    }, sort_keys=True).encode('ascii')
).hexdigest()
# The reviewed predecessor differs only in stage scheduling, not simulation.
_previous_sources = dict(SIMULATION_SOURCE_HASHES, **{
    'test_4_5seed.py': '5d5ed0f2e4ee5c840b5f0dab705b072d190c24295f016fc94758dfa7dff6a5d1',
})
PREVIOUS_SCHEDULING_CONTRACT_HASH = hashlib.sha256(json.dumps({
    'configuration': CONFIG_HASHES,
    'simulation_source': _previous_sources,
    'runtime_acceleration': runtime_acceleration_contract(),
}, sort_keys=True).encode('ascii')).hexdigest()


def resolve_trace(entry):
    path = (CONFIG_ROOT / entry['path']).resolve()
    if CONFIG_ROOT.resolve() not in path.parents:
        raise ValueError(f'Trace path is outside config directory: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'Trace SHA256 mismatch: {path}')
    return path


def case_trace_entries(kind, case_id):
    workload_id = case_id if kind == 'load' else FAULT_WORKLOAD_CASE['id']
    fault_id = LOAD_FAULT_CASE['id'] if kind == 'load' else case_id
    return (
        SCENARIO_MANIFEST['workloads'][workload_id],
        SCENARIO_MANIFEST['faults'][fault_id],
    )


def validate_fixed_configs():
    validate_per_satellite_computing_abilities()
    if COMMON_TEST.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported common test configuration schema')
    if TEST_SEED not in [int(seed) for seed in COMMON_TEST['scenario_seeds']]:
        raise ValueError('scenario_seed is not listed in common configuration')
    if int(SCENARIO_MANIFEST['scenario_seed']) != TEST_SEED:
        raise ValueError('Scenario manifest seed does not match scenario_seed')
    if int(SCENARIO_MANIFEST['duration']) != DURATION:
        raise ValueError('Scenario manifest duration does not match test duration')
    if COMMON_TEST['stream_contract'].get('mode') != 'fixed_trace_files':
        raise ValueError('Tests must use fixed trace files')
    if EXPECTED_TEST_STEP != 300:
        raise ValueError('The fixed test contract must contain exactly 300 steps')
    load_points = [int(item['target_load_tasks_per_second']) for item in LOAD_SETTINGS]
    if sorted(load_points) != list(range(100, 161, 10)):
        raise ValueError('Load experiment points must be 100 through 160 in steps of 10')
    if [float(item['fault_percent']) for item in FAULT_CASES] != [0, 3, 6, 9, 12, 15, 18]:
        raise ValueError('Unexpected fault experiment points')
    if abs(LOAD_FAULT_PROBABILITY - 0.18) > 1e-12:
        raise ValueError('Load experiments must use edge_p=0.18')
    if int(FAULT_LOAD_SETTING['target_load_tasks_per_second']) != 160:
        raise ValueError('Fault experiments must use load 160')
    fault_load = LOAD_SETTINGS_BY_POINT.get(int(FAULT_LOAD_SETTING['target_load_tasks_per_second']))
    if fault_load is None or any(
        float(FAULT_LOAD_SETTING[key]) != float(fault_load[key])
        for key in ('poisson_rate_g', 'poisson_rate_s')
    ):
        raise ValueError('Fault workload must match a configured load experiment point')
    if abs(float(LOAD_TESTS['edge_q']) - (1.0 - LOAD_FAULT_PROBABILITY)) > 1e-12:
        raise ValueError('Load experiment edge_q must equal 1-edge_p')
    for item in FAULT_CASES:
        if abs(float(item['edge_q']) - (1.0 - float(item['edge_p']))) > 1e-12:
            raise ValueError(f"Fault case {item['id']} has inconsistent edge_p/edge_q")
    for setting in LOAD_SETTINGS:
        workload, _ = case_trace_entries('load', setting['id'])
        resolve_trace(workload)
    for item in FAULT_CASES:
        workload, fault = case_trace_entries('fault', item['id'])
        resolve_trace(workload)
        resolve_trace(fault)
    resolve_trace(SCENARIO_MANIFEST['faults'][LOAD_FAULT_CASE['id']])


validate_fixed_configs()

MODEL_DIRS = {
    'D3QN': 'D3QN',
    'D3QN+': 'D3QN',
    'DDQN': 'DDQN',
    'PPO': 'ppo',
    'ISACR_self_elimination': 'ISACR_self_elimination',
    'ISACR_self_elimination+': 'ISACR_self_elimination',
}


def number_tag(value):
    return f'{value:g}'.replace('.', 'p')


def last_step(path):
    if not path.is_file():
        return None
    text = path.read_text(encoding='utf-8', errors='replace')
    matches = re.findall(r'^====== step (\d+) ======$', text, re.MULTILINE)
    return int(matches[-1]) if matches else None


def checkpoint_candidates(algorithm, seed):
    algorithm = MODEL_ALGORITHMS.get(algorithm, algorithm)
    directory = MODEL_ROOT / MODEL_DIRS[algorithm]
    if algorithm == 'PPO':
        paths = [
            path
            for path in directory.glob(
                f'PPO_seed{seed}_train_reward_best_reward*'
            )
            if path.is_dir()
            and (path / 'actor.pth').is_file()
            and (path / 'critic.pth').is_file()
        ]
    else:
        paths = list(directory.glob(
            f'{algorithm}_seed{seed}_train_reward_best_reward*.pth'
        ))

    def reward_value(path):
        match = re.search(r'_reward(-?\d+p\d+)', path.name)
        return float(match.group(1).replace('p', '.')) if match else float('-inf')

    return sorted(paths, key=reward_value, reverse=True)


def training_log_candidates(algorithm, seed, checkpoint):
    algorithm = MODEL_ALGORITHMS.get(algorithm, algorithm)
    if checkpoint.parent.name == f'seed{seed}':
        directory = checkpoint.parent
        return sorted(directory.glob(f'train_*seed{seed}*.txt'))
    directory = MODEL_ROOT / MODEL_DIRS[algorithm]
    return sorted(directory.glob(f'train_{algorithm}_seed{seed}_*.txt'))


def validate_model_training_config(config, algorithm, seed):
    if algorithm == 'PPO':
        # Imported PPO runs may store checkpoints and logs outside this suite.
        # Validate every training parameter while allowing those artifact paths.
        config = json.loads(json.dumps(config))
        expected = build_training_config(algorithm, seed)
        if config['agent'].get('initial_model_path') is not None:
            raise ValueError('PPO evaluation requires a model trained from scratch')
        if 'initial_model_path' not in expected['agent']:
            config['agent'].pop('initial_model_path', None)
        config['agent']['model_path'] = expected['agent']['model_path']
        config['environment']['save_training_data'] = expected['environment']['save_training_data']
    validate_training_config(config, algorithm, seed)


def find_ready_model(algorithm, seed):
    if algorithm in MODEL_ALGORITHMS:
        return find_ready_model(MODEL_ALGORITHMS[algorithm], seed)
    if algorithm == 'ISACR_self_elimination':
        from train_4_5seed import completed_result
        completed = completed_result(algorithm, seed)
        if completed is not None and completed.get('copied_from_coefficient') == 2.0:
            return Path(completed['reward_checkpoint']), Path(completed['training_log']), 'ready (copied coefficient 2.0)'
    result_path = (
        TRAINING_STATUS_ROOT / algorithm / f'seed{seed}' / 'process_result.json'
    )
    if not result_path.is_file():
        return None, None, 'training process result not found'
    unit_dir = result_path.parent
    try:
        result = json.loads(result_path.read_text(encoding='utf-8'))
        config = json.loads((unit_dir / 'config.json').read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError):
        return None, None, 'invalid training process result'
    checkpoint_text = result.get('reward_checkpoint')
    training_log_text = result.get('training_log')
    if (
        result.get('exit_code') != 0
        or result.get('final_step') != EXPECTED_TRAINING_STEP
        or config.get('general', {}).get('random_seed') != seed
        or config.get('general', {}).get('scenario_seed') != EXPECTED_TRAINING_SCENARIO_SEED
        or config.get('general', {}).get('duration') != EXPECTED_TRAINING_STEP
        or config.get('general', {}).get('training_reward_save_start_step') != EXPECTED_TRAINING_REWARD_SAVE_START_STEP
        or config.get('general', {}).get('model_saving_enabled') is not True
        or config.get('environment', {}).get('fault_fill_multiplier') != build_training_config(algorithm, seed)['environment']['fault_fill_multiplier']
        or config.get('environment', {}).get('edge_p') != EXPECTED_TRAINING_EDGE_P
        or config.get('environment', {}).get('edge_q') != EXPECTED_TRAINING_EDGE_Q
        or any(
            config.get('environment', {}).get(key) != value
            for key, value in EXPECTED_TRAINING_LOAD.items()
        )
        or not checkpoint_text
        or not training_log_text
    ):
        return None, None, 'training process result is incomplete'
    checkpoint = Path(checkpoint_text)
    training_log = Path(training_log_text)
    model_directory = MODEL_ROOT / MODEL_DIRS[algorithm]
    if MODEL_ROOT.resolve() != DEFAULT_MODEL_ROOT.resolve():
        checkpoint = model_directory / checkpoint.name
        training_log = model_directory / training_log.name
    checkpoint_variant = os.environ.get(
        'SATELLITE_CHECKPOINT_VARIANT', 'train_reward_best'
    )
    if checkpoint_variant == 'latest':
        latest_checkpoint_text = result.get('latest_checkpoint')
        if latest_checkpoint_text:
            checkpoint = Path(latest_checkpoint_text)
            if MODEL_ROOT.resolve() != DEFAULT_MODEL_ROOT.resolve():
                checkpoint = model_directory / checkpoint.name
        else:
            checkpoint = find_latest_checkpoint(model_directory / (
                f'PPO_seed{seed}'
                if algorithm == 'PPO'
                else f'{algorithm}_seed{seed}.pth'
            ))
    elif checkpoint_variant != 'train_reward_best':
        return None, None, (
            'unsupported SATELLITE_CHECKPOINT_VARIANT: '
            f'{checkpoint_variant}'
        )
    if checkpoint is None or not checkpoint.exists():
        return None, None, f'{checkpoint_variant} checkpoint not found'
    try:
        validate_model_training_config(config, algorithm, seed)
        checkpoint_config = load_checkpoint_training_config(checkpoint)
    except ValueError as error:
        return None, None, str(error)
    if any(checkpoint_config[section] != config[section]
           for section in ('general', 'agent', 'environment')):
        return None, None, 'checkpoint metadata differs from persisted training config'
    if last_step(training_log) != EXPECTED_TRAINING_STEP:
        return None, None, (
            f'training log is not complete at step {EXPECTED_TRAINING_STEP}'
        )
    return checkpoint, training_log, 'ready'


def live_training_manager():
    for lock_path in TRAINING_LOCKS:
        if not lock_path.is_file():
            continue
        try:
            process_id = int(lock_path.read_text(encoding='ascii').strip())
            if os.name == 'nt':
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(
                    0x1000, False, process_id
                )
                if not handle:
                    continue
                ctypes.windll.kernel32.CloseHandle(handle)
            else:
                os.kill(process_id, 0)
            return lock_path, process_id
        except (OSError, ValueError):
            continue
    return None, None


def checkpoint_hashes(checkpoint):
    if checkpoint.is_file():
        files = (checkpoint,)
    else:
        files = (checkpoint / 'actor.pth', checkpoint / 'critic.pth')
    files += (Path(str(checkpoint) + '.training.json'),)
    return {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def validate_ready_models(algorithms, seeds):
    models = []
    errors = {}
    for algorithm in algorithms:
        for seed in seeds:
            checkpoint, training_log, status = find_ready_model(algorithm, seed)
            key = f'{algorithm}/seed{seed}'
            if checkpoint is None or training_log is None:
                errors[key] = status
                continue
            models.append({
                'algorithm': algorithm,
                'training_seed': seed,
                'checkpoint': str(checkpoint.resolve()),
                'checkpoint_sha256': checkpoint_hashes(checkpoint),
                'training_log': str(training_log.resolve()),
                'training_final_step': last_step(training_log),
            })
    write_json(MODEL_MANIFEST_PATH, {
        'expected_models': len(algorithms) * len(seeds),
        'ready_models': len(models),
        'algorithms': algorithms,
        'training_seeds': seeds,
        'model_directories': MODEL_DIRS,
        'fixed_config_root': str(CONFIG_ROOT),
        'result_root': str(TEST_ROOT),
        'models': models,
        'model_errors': errors,
    })
    return models, errors


def unit_directory(algorithm, seed):
    return TEST_ROOT / algorithm / f'seed{seed}'


def case_directory(algorithm, seed, kind, value):
    if kind == 'load':
        name = f'load_{value["target_load_tasks_per_second"]}'
    else:
        name = f'fault_p{number_tag(value)}'
    return unit_directory(algorithm, seed) / kind / name


def build_test_config(algorithm, checkpoint, kind, value, raw_path):
    if kind == 'load':
        load_setting = value
        fault_probability = LOAD_FAULT_PROBABILITY
        case_id = value['id']
    else:
        load_setting = FAULT_LOAD_SETTING
        fault_probability = value
        case_id = FAULT_CASE_BY_PROBABILITY[fault_probability]['id']

    workload_trace, fault_trace = case_trace_entries(kind, case_id)
    workload_trace_path = resolve_trace(workload_trace)
    fault_trace_path = resolve_trace(fault_trace)

    config = build_test_config_base(algorithm, load_setting, checkpoint)
    config['general'].update(COMMON_TEST['general'])
    config['general'].update(
        random_seed=RANDOM_SEED,
        scenario_seed=TEST_SEED,
        print_cycle=1,
        show_test_summary=True,
    )
    config['agent']['model_path'] = str(checkpoint)
    config['environment'].update(COMMON_TEST['environment'])
    config['environment'].update(load_setting)
    config['environment'].update(
        id=FAULT_WORKLOAD_CASE['id'] if kind == 'fault' else case_id,
        edge_p=fault_probability,
        edge_q=1.0 - fault_probability,
        node_p=0.0,
        node_q=0.0,
        markov_mode=True,
        fault_compute_double_mean=True,
        fault_fill_multiplier=FAULT_FILL_MULTIPLIER,
        visualize=False,
        save_log=False,
        save_training_data=str(raw_path),
        workload_trace_path=str(workload_trace_path),
        workload_trace_sha256=workload_trace['sha256'],
        workload_trace_task_count=int(workload_trace['task_count']),
        fault_trace_path=str(fault_trace_path),
        fault_trace_sha256=fault_trace['sha256'],
        fault_trace_event_count=int(fault_trace['event_count']),
    )
    for key, expected in COMMON_TEST['general'].items():
        if config['general'].get(key) != expected:
            raise ValueError(f'Common general setting changed: {key}')
    for key, expected in COMMON_TEST['environment'].items():
        if key == 'computing_ability':
            continue
        if key in {'fault_fill_multiplier'} and float(config['environment'].get(key)) == float(expected):
            continue
        if config['environment'].get(key) != expected:
            raise ValueError(f'Common environment setting changed: {key}')
    config['test_contract'] = {
        'runtime_acceleration': runtime_acceleration_contract(),
        'case_id': case_id,
        'contract_sha256': TEST_CONTRACT_HASH,
        'configuration_sha256': CONFIG_HASHES,
        'simulation_source_sha256': SIMULATION_SOURCE_HASHES,
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': checkpoint_hashes(checkpoint),
        'python_hash_seed': PYTHON_HASH_SEED,
        'stream_contract': COMMON_TEST['stream_contract'],
        'workload_trace': workload_trace,
        'fault_trace': fault_trace,
    }
    if algorithm in DESTINATION_SELECTION_ALGORITHMS:
        config['general']['select_mode'] = 3
        config['test_contract']['destination_selection'] = {
            'method': 'masked_joint_destination_action_q',
            'model_algorithm': MODEL_ALGORITHMS[algorithm],
            'source_sha256': hashlib.sha256(
                (PROJECT_ROOT / 'test_config.py').read_bytes()
            ).hexdigest(),
        }
    return config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8',
    )


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def result_row(algorithm, seed, kind, value, checkpoint, raw_path):
    if kind == 'load':
        load_setting = value
        fault_probability = LOAD_FAULT_PROBABILITY
        case_id = value['id']
    else:
        load_setting = FAULT_LOAD_SETTING
        fault_probability = value
        case_id = FAULT_CASE_BY_PROBABILITY[fault_probability]['id']
    workload_trace, fault_trace = case_trace_entries(kind, case_id)
    row = parse_test_result(raw_path, algorithm, load_setting, DURATION)
    row.update(
        training_seed=seed,
        scenario_seed=TEST_SEED,
        fault_fill_multiplier=FAULT_FILL_MULTIPLIER,
        edge_p=fault_probability,
        edge_q=1.0 - fault_probability,
        test_type=kind,
        model_path=str(checkpoint),
        model_sha256=json.dumps(checkpoint_hashes(checkpoint), sort_keys=True),
        test_case_id=case_id,
        test_contract_sha256=TEST_CONTRACT_HASH,
        expected_workload_trace_sha256=workload_trace['sha256'],
        expected_fault_trace_sha256=fault_trace['sha256'],
    )
    if kind == 'fault':
        row['fault_percent'] = f'{fault_probability * 100:g}'
    return row


def complete_result(raw_path):
    return raw_path.is_file() and last_step(raw_path) == EXPECTED_TEST_STEP


def stored_contract_matches(config_path, algorithm=None):
    if not config_path.is_file():
        return False
    stored = json.loads(config_path.read_text(encoding='utf-8'))
    random_offloading_contract = stored.get('test_contract', {}).get('random_offloading')
    if (random_offloading_contract is not None
            or stored.get('test_contract', {}).get('algorithm') == 'D3QN-RO'):
        return False
    if stored.get('test_contract', {}).get('heuristic_offloading') is not None:
        return False
    if stored.get('test_contract', {}).get('local_computing') is not None:
        return False
    destination_selection = stored.get('test_contract', {}).get('destination_selection')
    if algorithm in DESTINATION_SELECTION_ALGORITHMS and (
        not isinstance(destination_selection, dict)
        or destination_selection.get('method') != 'masked_joint_destination_action_q'
        or destination_selection.get('model_algorithm') != MODEL_ALGORITHMS[algorithm]
        or stored.get('general', {}).get('select_mode') != 3
    ):
        return False
    if destination_selection is not None and (
        stored.get('general', {}).get('select_mode') != 3
        or destination_selection.get('source_sha256') != hashlib.sha256(
            (PROJECT_ROOT / 'test_config.py').read_bytes()
        ).hexdigest()
    ):
        return False
    return (
        stored.get('test_contract', {}).get('contract_sha256')
        in {TEST_CONTRACT_HASH, PREVIOUS_SCHEDULING_CONTRACT_HASH}
    )


def stored_checkpoint_matches(config_path, checkpoint):
    """Return whether a completed case used these exact checkpoint contents."""
    if not config_path.is_file():
        return False
    stored = json.loads(config_path.read_text(encoding='utf-8'))
    stored_path = stored.get('test_contract', {}).get('checkpoint')
    if stored_path is None:
        stored_path = stored.get('agent', {}).get('model_path')
    if not stored_path:
        return False
    return (
        Path(stored_path).resolve() == checkpoint.resolve()
        and stored.get('test_contract', {}).get('checkpoint_sha256')
        == checkpoint_hashes(checkpoint)
    )




SIMPLE_OFFLOADING_RULES = ('destination', 'local', 'random', 'random_idle', 'nearest', 'least_finish')


@contextmanager
def simple_offloading(rule, scenario_seed):
    import random
    import numpy as np
    import SatelliteNetworkSimulator_Computing as simulation
    if rule not in SIMPLE_OFFLOADING_RULES:
        raise ValueError(f'Unknown compute rule: {rule}')
    original = simulation.Satellite_with_Computing

    class SimpleSatellite(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.require_computed_before_downlink = True
            self._simple_packet = None
            self._simple_rng = random.Random(f'D3QN-simple:{rule}:{scenario_seed}:{self.name}')
            original_get = self.forward_queue.get

            def get_packet():
                event = original_get()
                event.callbacks.append(self._simple_receive)
                return event

            self.forward_queue.get = get_packet

        def _simple_receive(self, event):
            self._simple_packet = event.value
            packet = event.value
            if rule in ('random', 'random_idle') and not packet.information[0]:
                visit = (self.name, packet.hops)
                if getattr(packet, '_simple_random_visit', None) != visit:
                    packet._simple_random_visit = visit
                    packet._simple_random_compute = self._simple_rng.random() < 0.3

        def get_next_hop(self, current_state, destination, packet_size=None,
                         forced_action=None, prepared_policy_state=None,
                         prepared_action_mapping=None, *, hops, wait_count=0,
                         previous_hop=None, postcompute_previous_hop=None):
            if self.epsilon != 0:
                raise ValueError('Simple baselines require deterministic evaluation epsilon=0')
            mask = self.valid_action_mask(destination, packet_size, current_state[-1],
                                          hops, wait_count, previous_hop,
                                          postcompute_previous_hop)
            packet = self._simple_packet
            if forced_action is None and not current_state[-1]:
                # Physical action availability remains the shared environment's
                # responsibility, including CPU gating when no route exists.
                compute = destination == self.name or not mask[:4].any()
                if rule == 'local':
                    compute = True
                elif rule in ('random', 'random_idle'):
                    random_compute = bool(getattr(packet, '_simple_random_compute', False))
                    if rule == 'random_idle':
                        cpu_wait = max(0.0, self.computing_remain / self.computing_ability
                                       - self.is_computing * (
                                           self.env.now - self.last_computing_time))
                        random_compute &= cpu_wait <= simulation.LOCAL_COMPUTE_WAIT_THRESHOLD
                    compute |= random_compute
                elif rule in ('nearest', 'least_finish') and not compute:
                    target = getattr(packet, '_simple_compute_target', None)
                    if target is None or (target != self.name and target not in self.neighbors):
                        legal = np.flatnonzero(mask[:4]).tolist()
                        neighbors = getattr(self, 'action_neighbors', self.neighbors)
                        if rule == 'least_finish':
                            # Use received one-hop resource messages, not global CPU state.
                            demand = packet.information[2]
                            local_finish = max(0.0, self.computing_remain / self.computing_ability
                                               - self.is_computing * (
                                                   self.env.now - self.last_computing_time))
                            local_finish += demand / self.computing_ability
                            finish_times = {}
                            for index in legal:
                                neighbor = neighbors[index]
                                resource = self.neighbor_states[neighbor]
                                if resource[2] <= 0:
                                    continue
                                ability = np.sqrt(resource[2]) * simulation.MAX_COMPUTING_ABILITY
                                finish_times[index] = (
                                    resource[3] * simulation.CT_FAC + demand / ability
                                    + self._normalized_link_transmission_wait(neighbor)
                                    * simulation.TRANSMISSION_WAIT_TIME_SCALE
                                    + packet_size / self.transmission_rate
                                    + self.propagator.propagation_delays.get(
                                        (self.name, neighbor), float('inf'))
                                )
                            legal = list(finish_times)
                            if not legal or local_finish <= min(finish_times.values()):
                                compute = True
                                legal = []
                        if legal:
                            action = min(legal, key=lambda index: (
                                finish_times[index] if rule == 'least_finish' else
                                self.propagator.propagation_delays.get(
                                    (self.name, neighbors[index]), float('inf')),
                                neighbors[index],
                            ))
                            target = neighbors[action]
                            if packet is not None:
                                packet._simple_compute_target = target
                            forced_action = action
                        else:
                            compute = True
                    elif target == self.name:
                        compute = True
                    else:
                        neighbors = getattr(self, 'action_neighbors', self.neighbors)
                        action = neighbors.index(target)
                        if mask[action]:
                            forced_action = action
                        else:
                            # A temporarily invalid target falls back to the
                            # shared physical lifecycle, not a fabricated action.
                            compute = True
                if compute:
                    if not mask[4]:
                        return None
                    forced_action = 4
            if forced_action is not None:
                if not mask[int(forced_action)]:
                    return None
                return super().get_next_hop(
                    current_state, destination, packet_size, forced_action,
                    prepared_policy_state, prepared_action_mapping, hops=hops,
                    wait_count=wait_count, previous_hop=previous_hop,
                    postcompute_previous_hop=postcompute_previous_hop,
                )
            original_q = self.q_net

            def routing_q(state, action_mask):
                routing_mask = np.asarray(action_mask, dtype=bool).copy()
                routing_mask[4] = False
                values = original_q(state, routing_mask).clone()
                values[..., 4] = float('-inf')
                return values

            self.q_net = routing_q
            try:
                return super().get_next_hop(
                    current_state, destination, packet_size, None,
                    prepared_policy_state, prepared_action_mapping, hops=hops,
                    wait_count=wait_count, previous_hop=previous_hop,
                    postcompute_previous_hop=postcompute_previous_hop,
                )
            finally:
                self.q_net = original_q

    simulation.Satellite_with_Computing = SimpleSatellite
    try:
        yield
    finally:
        simulation.Satellite_with_Computing = original


RANDOM_OFFLOADING_CONTRACT = {
    'method': 'random_compute_then_d3qn_routing',
    'model_algorithm': 'D3QN',
    'compute_probability': 0.3,
    'masked_compute_features': ['a', 'c', 'g'],
    'draw_once_per_visit': True,
    'include_upload_satellite': True,
    'destination_compute_fallback': True,
    'no_forward_compute_fallback': True,
    'no_forward_cpu_wait_gate': False,
    'require_computed_before_downlink': True,
}


@contextmanager
def random_offloading(scenario_seed, *, no_forward_compute_fallback=True):
    """Random compute placement; resource-blind D3QN chooses forwarding only."""
    import random
    import numpy as np
    import SatelliteNetworkSimulator_Computing as simulation

    original = simulation.Satellite_with_Computing

    class RandomOffloadingSatellite(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.require_computed_before_downlink = True
            self._ro_rng = random.Random(f'D3QN-RO:{scenario_seed}:{self.name}')
            self._ro_compute = False
            original_get = self.forward_queue.get

            def get_packet():
                event = original_get()
                event.callbacks.append(self._ro_receive_packet)
                return event

            self.forward_queue.get = get_packet

        def _ro_receive_packet(self, event):
            packet = event.value
            self._ro_compute = False
            if packet.information[0]:
                return
            visit = (self.name, packet.hops)
            if getattr(packet, '_ro_visit', None) != visit:
                packet._ro_visit = visit
                packet._ro_compute = (
                    packet.destination == self.name or self._ro_rng.random() < 0.3
                )
            self._ro_compute = packet._ro_compute

        def get_current_state(self, destination, hops, is_computed, mission_state):
            state = super().get_current_state(destination, hops, is_computed, mission_state)
            # Pure_DQN: four 8-field neighbor slots, then five local fields.
            for offset in range(0, 32, 8):
                for index in (2, 3, 5):
                    state[offset + index] = 0.0
            state[34:37] = [0.0, 0.0, 0.0]
            return state

        def valid_action_mask(self, destination, packet_size, is_computed, hops,
                              wait_count=0, previous_hop=None,
                              postcompute_previous_hop=None):
            mask = super().valid_action_mask(
                destination, packet_size, is_computed, hops, wait_count,
                previous_hop, postcompute_previous_hop,
            )
            if not is_computed:
                # RO falls back to the local CPU queue without the wait gate.
                # Keep the common hop limit, and inspect neighbors before masking them.
                if (no_forward_compute_fallback and not mask[:4].any()
                        and (destination == self.name or hops < 2 * self.max_hop)):
                    mask[4] = True
                if destination == self.name or self._ro_compute:
                    mask[:4] = False
                elif mask[:4].any() or not no_forward_compute_fallback:
                    mask[4] = False
            return mask

        def get_next_hop(self, current_state, destination, packet_size=None,
                         forced_action=None, prepared_policy_state=None,
                         prepared_action_mapping=None, *, hops, wait_count=0,
                         previous_hop=None, postcompute_previous_hop=None):
            mask = self.valid_action_mask(
                destination, packet_size, current_state[-1], hops, wait_count,
                previous_hop, postcompute_previous_hop,
            )
            if not current_state[-1] and (
                    destination == self.name or self._ro_compute
                    or (no_forward_compute_fallback and not mask[:4].any())):
                if not mask[4]:
                    return None
                return super().get_next_hop(
                    current_state, destination, packet_size, 4,
                    prepared_policy_state, prepared_action_mapping, hops=hops,
                    wait_count=wait_count, previous_hop=previous_hop,
                    postcompute_previous_hop=postcompute_previous_hop,
                )
            original_q = self.q_net

            def routing_q(state, action_mask):
                mask = np.asarray(action_mask, dtype=bool).copy()
                mask[4] = False
                values = original_q(state, mask).clone()
                values[..., 4] = float('-inf')
                return values

            # Base selection adds compute action 4 for an uncomputed packet.
            # Exclude it from dueling aggregation as well as final argmax.
            self.q_net = routing_q
            try:
                return super().get_next_hop(
                    current_state, destination, packet_size, forced_action,
                    prepared_policy_state, prepared_action_mapping, hops=hops,
                    wait_count=wait_count, previous_hop=previous_hop,
                    postcompute_previous_hop=postcompute_previous_hop,
                )
            finally:
                self.q_net = original_q

    simulation.Satellite_with_Computing = RandomOffloadingSatellite
    try:
        yield
    finally:
        simulation.Satellite_with_Computing = original

def run_case(algorithm, seed, checkpoint, kind, value, force_rerun=False):
    directory = case_directory(algorithm, seed, kind, value)
    raw_path = directory / 'result.txt'
    running_path = directory / 'result.running.txt'
    config_path = directory / 'config.json'
    summary_path = directory / 'summary.json'
    directory.mkdir(parents=True, exist_ok=True)

    if kind == 'fault' and float(value) == LOAD_FAULT_PROBABILITY:
        # This physical scenario is also the load-scan intersection.
        shared_load = dict(FAULT_LOAD_SETTING, id=FAULT_WORKLOAD_CASE['id'])
        shared = run_case(algorithm, seed, checkpoint, 'load', shared_load, force_rerun)
        archive_test_outputs((raw_path, running_path, config_path, summary_path))
        shutil.copy2(shared['raw_result_file'], raw_path)
        config = build_test_config(algorithm, checkpoint, kind, value, raw_path)
        write_json(config_path, config)
        row = result_row(algorithm, seed, kind, value, checkpoint, raw_path)
        write_json(summary_path, row)
        return row

    if not force_rerun and complete_result(raw_path):
        try:
            reusable = (
                stored_contract_matches(config_path, algorithm)
                and stored_checkpoint_matches(config_path, checkpoint)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            reusable = False
        if reusable:
            running_path.unlink(missing_ok=True)
            if summary_path.is_file():
                return json.loads(summary_path.read_text(encoding='utf-8'))
            row = result_row(algorithm, seed, kind, value, checkpoint, raw_path)
            write_json(summary_path, row)
            return row
        # A complete result with a stale contract, malformed metadata, or an
        # older checkpoint is rerun so this batch uses the current contract.

    config = build_test_config(algorithm, checkpoint, kind, value, running_path)
    prepare_test_acceleration(config)
    archive_test_outputs((raw_path, running_path, config_path, summary_path))
    write_json(config_path, config)
    if algorithm in DESTINATION_SELECTION_ALGORITHMS:
        with destination_selection():
            run_training(config)
    else:
        run_training(config)
    if not complete_result(running_path):
        raise RuntimeError(
            f'Incomplete test result: {running_path}, '
            f'expected final step {EXPECTED_TEST_STEP}, got {last_step(running_path)}'
        )

    running_path.replace(raw_path)
    config['environment']['save_training_data'] = str(raw_path)
    write_json(config_path, config)
    row = result_row(algorithm, seed, kind, value, checkpoint, raw_path)
    write_json(summary_path, row)
    return row


def case_key(kind, value):
    if kind == 'load':
        return ('load', int(value['target_load_tasks_per_second']))
    return ('fault', float(value))


def reusable_case(algorithm, seed, checkpoint, kind, value, force_rerun):
    """Return whether a completed result can be kept for this checkpoint."""
    if force_rerun:
        return False
    directory = case_directory(algorithm, seed, kind, value)
    try:
        return (
            complete_result(directory / 'result.txt')
            and stored_contract_matches(directory / 'config.json', algorithm)
            and stored_checkpoint_matches(directory / 'config.json', checkpoint)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def row_key(row):
    if row.get('test_type') == 'load' or row.get('fault_percent') in (None, ''):
        return ('load', int(float(row['target_load_tasks_per_second'])))
    return ('fault', float(row['fault_percent']) / 100.0)


def count_selected_cases(rows, selected_case_keys=None):
    if selected_case_keys is None:
        return len(rows)
    return sum(row_key(row) in selected_case_keys for row in rows)


def save_unit_status(
    algorithm, seed, checkpoint, log_path, rows, errors,
    selected_case_keys=None,
):
    expected = (
        len(LOAD_SETTINGS) + len(FAULT_PROBABILITIES)
        if selected_case_keys is None else len(selected_case_keys)
    )
    completed = count_selected_cases(rows, selected_case_keys)
    status = {
        'algorithm': algorithm,
        'training_seed': seed,
        'model_path': str(checkpoint) if checkpoint else None,
        'training_log': str(log_path) if log_path else None,
        'expected_cases': expected,
        'completed_cases': completed,
        'missing_cases': expected - completed,
        'errors': errors,
        'status': 'complete' if completed == expected and not errors else 'partial',
        'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
    }
    write_json(unit_directory(algorithm, seed) / 'STATUS.json', status)


def run_unit(job):
    (
        algorithm, seed, selected_load_settings,
        selected_fault_probabilities, force_rerun,
    ) = job
    unit_dir = unit_directory(algorithm, seed)
    unit_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, log_path, reason = find_ready_model(algorithm, seed)
    if checkpoint is None:
        save_unit_status(
            algorithm, seed, None, None, [], [reason],
            selected_case_keys={
                ('load', int(item['target_load_tasks_per_second']))
                for item in selected_load_settings
            } | {
                ('fault', float(value))
                for value in selected_fault_probabilities
            },
        )
        return {
            'algorithm': algorithm,
            'training_seed': seed,
            'status': 'not_ready',
            'reason': reason,
        }

    load_csv = unit_dir / 'load' / 'load_test_results.csv'
    fault_csv = unit_dir / 'fault' / 'fault_test_results.csv'
    load_rows = read_rows(load_csv)
    fault_rows = read_rows(fault_csv)
    errors = []

    jobs = [
        ('load', value) for value in selected_load_settings
    ] + [
        ('fault', value) for value in selected_fault_probabilities
    ]
    selected_case_keys = {
        ('load', int(item['target_load_tasks_per_second']))
        for item in selected_load_settings
    } | {
        ('fault', float(value))
        for value in selected_fault_probabilities
    }
    load_rows[:] = [
        row for row in load_rows
        if ('load', int(float(row['target_load_tasks_per_second'])))
        not in selected_case_keys
    ]
    fault_rows[:] = [
        row for row in fault_rows
        if ('fault', float(row['fault_percent']) / 100.0) not in selected_case_keys
    ]
    write_rows(load_csv, load_rows)
    write_rows(fault_csv, fault_rows)
    for kind, value in jobs:
        try:
            reuse_intersection = (
                kind == 'fault' and float(value) == LOAD_FAULT_PROBABILITY
                and case_key('load', FAULT_LOAD_SETTING) in selected_case_keys
            )
            row = run_case(
                algorithm, seed, checkpoint, kind, value,
                force_rerun=force_rerun and not reuse_intersection,
            )
            target = load_rows if kind == 'load' else fault_rows
            key = (
                int(float(row['target_load_tasks_per_second']))
                if kind == 'load'
                else float(row['fault_percent'])
            )
            target[:] = [
                item for item in target
                if (
                    int(float(item['target_load_tasks_per_second']))
                    if kind == 'load'
                    else float(item['fault_percent'])
                ) != key
            ]
            target.append(row)
            write_rows(load_csv, load_rows)
            write_rows(fault_csv, fault_rows)
            save_unit_status(
                algorithm, seed, checkpoint, log_path,
                load_rows + fault_rows, errors,
                selected_case_keys,
            )
        except Exception as error:
            errors.append(f'{kind}:{value}: {error}')
            save_unit_status(
                algorithm, seed, checkpoint, log_path,
                load_rows + fault_rows, errors,
                selected_case_keys,
            )

    completed = count_selected_cases(load_rows + fault_rows, selected_case_keys)
    save_unit_status(
        algorithm, seed, checkpoint, log_path,
        load_rows + fault_rows, errors,
        selected_case_keys,
    )
    return {
        'algorithm': algorithm,
        'training_seed': seed,
        'status': 'complete' if completed == len(jobs) and not errors else 'partial',
        'completed_cases': completed,
        'expected_cases': len(jobs),
        'errors': errors,
    }


def run_case_job(job):
    """Run one independent load/fault case for case-level parallelism."""
    algorithm, seed, checkpoint, kind, value, force_rerun = job
    try:
        row = run_case(
            algorithm, seed, checkpoint, kind, value,
            force_rerun=force_rerun,
        )
        return {
            'algorithm': algorithm,
            'training_seed': seed,
            'kind': kind,
            'value': value,
            'status': 'complete',
            'row': row,
        }
    except Exception as error:
        return {
            'algorithm': algorithm,
            'training_seed': seed,
            'kind': kind,
            'value': value,
            'status': 'error',
            'error': f'{type(error).__name__}: {error}',
        }


def run_cases_parallel(algorithms, seeds, selected_load_settings,
                       selected_fault_probabilities, force_rerun, workers):
    """Run selected cases independently, allowing all ten workers to fill."""
    model_map = {}
    for algorithm in algorithms:
        for seed in seeds:
            checkpoint, log_path, reason = find_ready_model(algorithm, seed)
            if checkpoint is None:
                raise RuntimeError(f'{algorithm}/seed{seed}: {reason}')
            model_map[(algorithm, seed)] = (checkpoint, log_path)

    selected_case_keys = {
        ('load', int(item['target_load_tasks_per_second']))
        for item in selected_load_settings
    } | {
        ('fault', float(value))
        for value in selected_fault_probabilities
    }
    unit_data = {}
    pending_by_unit = {}
    for algorithm in algorithms:
        for seed in seeds:
            checkpoint, log_path = model_map[(algorithm, seed)]
            unit_dir = unit_directory(algorithm, seed)
            load_csv = unit_dir / 'load' / 'load_test_results.csv'
            fault_csv = unit_dir / 'fault' / 'fault_test_results.csv'
            load_rows = read_rows(load_csv)
            fault_rows = read_rows(fault_csv)
            existing_keys = {row_key(row) for row in load_rows + fault_rows}
            pending = set()
            for value in selected_load_settings:
                key = case_key('load', value)
                if (
                    key not in existing_keys
                    or not reusable_case(
                        algorithm, seed, checkpoint, 'load', value, force_rerun,
                    )
                ):
                    pending.add(key)
            for value in selected_fault_probabilities:
                key = case_key('fault', value)
                if (
                    key not in existing_keys
                    or not reusable_case(
                        algorithm, seed, checkpoint, 'fault', value, force_rerun,
                    )
                ):
                    pending.add(key)
            pending_by_unit[(algorithm, seed)] = pending
            # Remove only stale rows for cases that will actually be rerun.
            load_rows = [row for row in load_rows if row_key(row) not in pending]
            fault_rows = [row for row in fault_rows if row_key(row) not in pending]
            if load_rows:
                write_rows(load_csv, load_rows)
            else:
                load_csv.unlink(missing_ok=True)
            if fault_rows:
                write_rows(fault_csv, fault_rows)
            else:
                fault_csv.unlink(missing_ok=True)
            unit_data[(algorithm, seed)] = {
                'checkpoint': checkpoint,
                'log_path': log_path,
                'load_rows': load_rows,
                'fault_rows': fault_rows,
                'errors': [],
            }

    jobs = []
    for algorithm in algorithms:
        for seed in seeds:
            checkpoint = model_map[(algorithm, seed)][0]
            pending = pending_by_unit[(algorithm, seed)]
            shared_load_key = case_key('load', FAULT_LOAD_SETTING)
            shared_fault_key = case_key('fault', LOAD_FAULT_PROBABILITY)
            share_intersection = (
                shared_load_key in selected_case_keys
                and shared_fault_key in selected_case_keys
            )
            if share_intersection and shared_fault_key in pending:
                pending.add(shared_load_key)
            jobs.extend(
                (algorithm, seed, checkpoint, 'load', value, force_rerun)
                for value in selected_load_settings
                if case_key('load', value) in pending
            )
            jobs.extend(
                (algorithm, seed, checkpoint, 'fault', value, force_rerun)
                for value in selected_fault_probabilities
                if case_key('fault', value) in pending
                and not (share_intersection and float(value) == LOAD_FAULT_PROBABILITY)
            )
    statuses = []
    for (algorithm, seed), data in unit_data.items():
        save_unit_status(
            algorithm, seed, data['checkpoint'], data['log_path'],
            data['load_rows'] + data['fault_rows'], data['errors'],
            selected_case_keys,
        )
    if jobs:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            futures = {executor.submit(run_case_job, job): job for job in jobs}
            for future in as_completed(futures):
                outcome = future.result()
                key = (outcome['algorithm'], outcome['training_seed'])
                data = unit_data[key]
                if outcome['status'] == 'complete':
                    rows = [outcome['row']]
                    if (outcome['kind'] == 'load'
                            and case_key('load', outcome['value']) == case_key('load', FAULT_LOAD_SETTING)
                            and case_key('fault', LOAD_FAULT_PROBABILITY) in selected_case_keys):
                        # The load job is complete; publish the same run to the
                        # fault scan without submitting another simulation.
                        rows.append(run_case(
                            outcome['algorithm'], outcome['training_seed'],
                            data['checkpoint'], 'fault', LOAD_FAULT_PROBABILITY,
                            force_rerun=False,
                        ))
                    for row in rows:
                        target = data['load_rows'] if row_key(row)[0] == 'load' else data['fault_rows']
                        target[:] = [item for item in target if row_key(item) != row_key(row)]
                        target.append(row)
                    unit_dir = unit_directory(*key)
                    write_rows(unit_dir / 'load' / 'load_test_results.csv', data['load_rows'])
                    write_rows(unit_dir / 'fault' / 'fault_test_results.csv', data['fault_rows'])
                else:
                    data['errors'].append(
                        f"{outcome['kind']}:{outcome['value']}: {outcome['error']}"
                    )
                save_unit_status(
                    outcome['algorithm'], outcome['training_seed'],
                    data['checkpoint'], data['log_path'],
                    data['load_rows'] + data['fault_rows'], data['errors'],
                    selected_case_keys,
                )
                print(json.dumps(outcome, ensure_ascii=False, default=str), flush=True)

    for (algorithm, seed), data in unit_data.items():
        completed = count_selected_cases(
            data['load_rows'] + data['fault_rows'], selected_case_keys,
        )
        statuses.append({
            'algorithm': algorithm,
            'training_seed': seed,
            'status': (
                'complete'
                if completed == len(selected_case_keys)
                and not data['errors'] else 'partial'
            ),
            'completed_cases': completed,
            'expected_cases': len(selected_case_keys),
            'errors': data['errors'],
        })
    return statuses


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_rows(rows, group_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, '') for field in group_fields)].append(row)
    output = []
    # CSV-derived keys can mix numeric values (e.g. fault_percent) with
    # strings (e.g. an empty/missing value).  Give numeric keys a stable
    # numeric ordering and fall back to text for non-numeric keys so that
    # incomplete rows do not make aggregation fail in sorted().
    def sort_key(item):
        key, _ = item
        return tuple(
            (0, numeric(value)) if numeric(value) is not None
            else (1, str(value))
            for value in key
        )

    for key, items in sorted(groups.items(), key=sort_key):
        result = dict(zip(group_fields, key))
        result['seed_count'] = len({item.get('training_seed') for item in items})
        fields = set().union(*(item.keys() for item in items))
        for field in sorted(fields):
            if field in group_fields or field in {
                'training_seed', 'scenario_seed', 'model_path',
                'raw_result_file', 'test_type',
            }:
                continue
            values = [numeric(item.get(field)) for item in items]
            values = [value for value in values if value is not None]
            if values:
                result[field] = sum(values) / len(values)
        output.append(result)
    return output


def fairness_audit(rows, algorithms=None):
    algorithms = ALGORITHMS if algorithms is None else algorithms
    groups = defaultdict(list)
    for row in rows:
        case_id = row.get('test_case_id')
        if not case_id:
            if row.get('test_type') == 'fault' or row.get('fault_percent') not in (None, ''):
                case_id = f"fault_{row.get('fault_percent')}"
            else:
                case_id = f"load_{row.get('target_load_tasks_per_second')}"
        groups[case_id].append(row)

    cases = []
    for case_id, items in sorted(groups.items()):
        workload_hashes = {
            item.get('workload_trace_sha256') for item in items
            if item.get('workload_trace_sha256')
        }
        fault_hashes = {
            item.get('fault_trace_sha256') for item in items
            if item.get('fault_trace_sha256')
        }
        expected_workload_hashes = {
            item.get('expected_workload_trace_sha256') for item in items
            if item.get('expected_workload_trace_sha256')
        }
        expected_fault_hashes = {
            item.get('expected_fault_trace_sha256') for item in items
            if item.get('expected_fault_trace_sha256')
        }
        contract_hashes = {
            item.get('test_contract_sha256') for item in items
            if item.get('test_contract_sha256')
        }
        verified = (
            len(items) == len(algorithms) * len(SEEDS)
            and {(item.get('algorithm'), int(item['training_seed'])) for item in items}
            == {(algorithm, seed) for algorithm in algorithms for seed in SEEDS}
            and all(
                item.get('workload_trace_sha256') == item.get('expected_workload_trace_sha256')
                and item.get('fault_trace_sha256') == item.get('expected_fault_trace_sha256')
                and item.get('test_contract_sha256') == TEST_CONTRACT_HASH
                for item in items
            )
            and len(workload_hashes) == 1
            and len(fault_hashes) == 1
            and workload_hashes == expected_workload_hashes
            and fault_hashes == expected_fault_hashes
            and contract_hashes == {TEST_CONTRACT_HASH}
        )
        cases.append({
            'case_id': case_id,
            'result_count': len(items),
            'workload_trace_count': len(workload_hashes),
            'fault_trace_count': len(fault_hashes),
            'expected_workload_trace_count': len(expected_workload_hashes),
            'expected_fault_trace_count': len(expected_fault_hashes),
            'contract_count': len(contract_hashes),
            'status': 'verified' if verified else 'unverified_or_mismatch',
        })
    return {
        'status': (
            'verified'
            if len(cases) == len(LOAD_SETTINGS) + len(FAULT_CASES)
            and all(item['status'] == 'verified' for item in cases)
            else 'unverified_or_mismatch'
        ),
        'cases': cases,
    }


def current_result_rows(rows, algorithm, seed):
    models = {}
    current = {}
    for row in rows:
        if (
            row.get('test_contract_sha256') != TEST_CONTRACT_HASH
            or row.get('algorithm') != algorithm
            or int(row['training_seed']) != seed
        ):
            continue
        model_path = row.get('model_path')
        if not model_path or not Path(model_path).exists():
            continue
        if model_path not in models:
            models[model_path] = json.dumps(checkpoint_hashes(Path(model_path)), sort_keys=True)
        if row.get('model_sha256') == models[model_path]:
            current[row['test_case_id']] = row
    return list(current.values())


def save_global_summaries(statuses, algorithms=None):
    algorithms = ALGORITHMS if algorithms is None else algorithms
    load_rows = []
    fault_rows = []
    configured_loads = {
        int(item['target_load_tasks_per_second']) for item in LOAD_SETTINGS
    }
    configured_faults = {
        float(probability * 100) for probability in FAULT_PROBABILITIES
    }
    for algorithm in algorithms:
        for seed in SEEDS:
            unit_dir = unit_directory(algorithm, seed)
            load_rows.extend(
                row for row in current_result_rows(read_rows(
                    unit_dir / 'load' / 'load_test_results.csv'
                ), algorithm, seed)
                if int(float(row['target_load_tasks_per_second']))
                in configured_loads
                and row.get('test_type') == 'load'
            )
            fault_rows.extend(
                row for row in current_result_rows(read_rows(
                    unit_dir / 'fault' / 'fault_test_results.csv'
                ), algorithm, seed)
                if float(row['fault_percent']) in configured_faults
                and row.get('test_type') == 'fault'
            )
    write_rows(TEST_ROOT / 'all_load_results.csv', load_rows)
    write_rows(TEST_ROOT / 'all_fault_results.csv', fault_rows)
    write_rows(
        TEST_ROOT / 'five_seed_load_mean.csv',
        mean_rows(load_rows, ('algorithm', 'target_load_tasks_per_second')),
    )
    write_rows(
        TEST_ROOT / 'five_seed_fault_mean.csv',
        mean_rows(fault_rows, ('algorithm', 'fault_percent')),
    )
    audit = fairness_audit(load_rows + fault_rows, algorithms)
    expected_cases = len(algorithms) * len(SEEDS) * (
        len(LOAD_SETTINGS) + len(FAULT_PROBABILITIES)
    )
    completed_cases = len(load_rows) + len(fault_rows)
    complete = (
        completed_cases == expected_cases
        and audit['status'] == 'verified'
        and all(item.get('status') == 'complete' for item in statuses)
    )
    write_json(
        TEST_ROOT / 'TEST_STATUS.json',
        {
            'status': 'complete' if complete else 'partial',
            'expected_cases': expected_cases,
            'completed_cases': completed_cases,
            'algorithms': algorithms,
            'training_seeds': SEEDS,
            'load_points': [item['target_load_tasks_per_second'] for item in LOAD_SETTINGS],
            'load_fault_probability': LOAD_FAULT_PROBABILITY,
            'fault_load': FAULT_LOAD_SETTING,
            'fault_percentages': [p * 100 for p in FAULT_PROBABILITIES],
            'duration_seconds': DURATION,
            'test_scenario_seed': TEST_SEED,
            'test_random_seed': RANDOM_SEED,
            'python_hash_seed': PYTHON_HASH_SEED,
            'test_contract_sha256': TEST_CONTRACT_HASH,
            'configuration_sha256': CONFIG_HASHES,
            'fairness_audit': audit,
            'max_workers': MAX_WORKERS,
            'statuses': statuses,
            'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        },
    )


def acquire_lock():
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
    except FileExistsError:
        return False
    with os.fdopen(descriptor, 'w', encoding='ascii') as stream:
        stream.write(f'pid={os.getpid()}\n')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_acceleration_argument(parser)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    algorithm_group = parser.add_mutually_exclusive_group()
    algorithm_group.add_argument('--algorithm', choices=TEST_ALGORITHMS)
    algorithm_group.add_argument('--algorithms', nargs='+', choices=TEST_ALGORITHMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument(
        '--isacr-plus', action=argparse.BooleanOptionalAction, default=True,
        help='Replace selected ISACR with joint destination/action selection; reuse its weights.',
    )
    parser.add_argument(
        '--experiment', choices=('both', 'load', 'fault'), default='both',
        help=(
            'Run both experiments, load only, or fault only. '
            'Load: 160 down to 100 tasks/s at edge_p=0.18. '
            'Fault: 18 percent down to 0 percent at load 160 tasks/s.'
        ),
    )
    parser.add_argument(
        '--load-point', type=int,
        choices=[int(item['target_load_tasks_per_second']) for item in LOAD_SETTINGS],
    )
    parser.add_argument(
        '--fault-percent', type=int,
        choices=[int(round(value * 100)) for value in FAULT_PROBABILITIES],
    )
    parser.add_argument('--status-only', action='store_true')
    parser.add_argument(
        '--allow-concurrent-training', action='store_true',
        help='Allow explicitly authorized tests of completed models while other training runs.',
    )
    parser.add_argument(
        '--rerun', action='store_true',
        help='Rerun every selected case even when a complete result exists.',
    )
    parser.add_argument(
        '--case-parallel', action='store_true',
        help='Schedule individual cases across workers instead of one job per seed.',
    )
    args = parser.parse_args()
    configure_acceleration(args.acceleration == 'on')
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f'--workers must be between 1 and {MAX_WORKERS}')

    if args.algorithms:
        algorithms = tuple(args.algorithms)
    elif args.algorithm:
        algorithms = (args.algorithm,)
    else:
        algorithms = DEFAULT_ALGORITHMS
    if args.isacr_plus:
        algorithms = tuple(dict.fromkeys(
            'ISACR_self_elimination+' if name == 'ISACR_self_elimination' else name
            for name in algorithms
        ))
    seeds = (args.seed,) if args.seed is not None else SEEDS
    if args.experiment == 'both' and args.load_point is None:
        # A complete load+fault run must cover every configured load point.
        selected_load_settings = tuple(
            LOAD_SETTINGS_BY_POINT[point] for point in LOAD_TEST_SEQUENCE
        )
    elif args.experiment == 'load' and args.load_point is None:
        selected_load_settings = tuple(
            LOAD_SETTINGS_BY_POINT[point] for point in LOAD_TEST_SEQUENCE
        )
    else:
        selected_load_settings = tuple(
            item for item in LOAD_SETTINGS
            if args.load_point is None
            or int(item['target_load_tasks_per_second']) == args.load_point
        )
    if args.experiment == 'fault':
        if args.load_point is not None:
            parser.error('--load-point cannot be used with --experiment fault')
        selected_load_settings = ()
        selected_fault_probabilities = tuple(
            probability for probability in FAULT_TEST_SEQUENCE
            if args.fault_percent is None
            or int(round(probability * 100)) == args.fault_percent
        )
    elif args.experiment == 'load' or args.load_point is not None:
        selected_fault_probabilities = ()
    else:
        selected_fault_probabilities = tuple(
            probability for probability in FAULT_TEST_SEQUENCE
            if args.fault_percent is None
            or int(round(probability * 100)) == args.fault_percent
        )
    if args.experiment == 'load' and args.fault_percent is not None:
        parser.error('--fault-percent cannot be used with --experiment load')
    if args.experiment == 'fault' and args.fault_percent is not None:
        selected_fault_probabilities = tuple(
            probability for probability in FAULT_PROBABILITIES
            if int(round(probability * 100)) == args.fault_percent
        )
    jobs = [
        (
            algorithm, seed, selected_load_settings,
            selected_fault_probabilities, args.rerun,
        )
        for algorithm in algorithms for seed in seeds
    ]

    if args.status_only:
        save_global_summaries([], algorithms)
        return
    training_lock, training_pid = live_training_manager()
    if training_pid is not None and not args.allow_concurrent_training:
        parser.error(
            f'training is still running: pid={training_pid}, '
            f'lock={training_lock}'
        )
    ready_models, model_errors = validate_ready_models(algorithms, seeds)
    expected_models = len(algorithms) * len(seeds)
    if len(ready_models) != expected_models or model_errors:
        parser.error(
            f'model validation failed: {len(ready_models)}/'
            f'{expected_models} ready; errors={model_errors}'
        )
    if not acquire_lock():
        print(f'TEST_STATUS=busy: {LOCK_PATH}', flush=True)
        return

    statuses = []
    try:
        staged_alternating = (
            not args.case_parallel
            and args.experiment == 'both'
            and args.load_point is None
            and args.fault_percent is None
            and len(selected_load_settings) == len(selected_fault_probabilities)
        )
        staged_load = not args.case_parallel and args.experiment == 'load' and args.load_point is None
        if staged_alternating:
            if len(selected_load_settings) != len(selected_fault_probabilities):
                raise RuntimeError(
                    'Alternating descending tests require equal load and fault point counts.'
                )
            # Run the shared intersection first, then alternate load and fault
            # stages. Finish every selected model at one stage before the next.
            stages = [(selected_load_settings[:1], selected_fault_probabilities[:1])]
            for load_setting, fault_probability in zip(
                selected_load_settings[1:],
                selected_fault_probabilities[1:],
            ):
                stages.extend([((load_setting,), ()), ((), (fault_probability,))])
            for stage_loads, stage_faults in stages:
                stage_statuses = run_cases_parallel(
                    algorithms, seeds, stage_loads, stage_faults,
                    args.rerun, args.workers,
                )
                statuses.extend(stage_statuses)
                if any(status['status'] != 'complete' for status in stage_statuses):
                    raise RuntimeError('Test stage did not complete')
        elif staged_load:
            # Synchronize by load point: all selected algorithm/seed cases at one
            # load must finish before the next lower load is submitted.
            for load_setting in selected_load_settings:
                statuses.extend(
                    run_cases_parallel(
                        algorithms, seeds, (load_setting,), (),
                        args.rerun, args.workers,
                    )
                )
        elif args.case_parallel:
            statuses = run_cases_parallel(
                algorithms, seeds, selected_load_settings,
                selected_fault_probabilities, args.rerun, args.workers,
            )
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
                futures = {executor.submit(run_unit, job): job for job in jobs}
                for future in as_completed(futures):
                    status = future.result()
                    statuses.append(status)
                    print(json.dumps(status, ensure_ascii=False), flush=True)
        save_global_summaries(statuses, algorithms)
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
