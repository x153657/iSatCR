"""Run the five retained ablations on fixed load and fault traces."""

import argparse
from test_config import (
    add_acceleration_argument, configure_acceleration, prepare_test_acceleration, run_training,
    destination_selection,
)
import ast
import copy
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from ablation.training_entry import VARIANT_NAMES, TRAINING_SEEDS, _json_ready, build_config as build_ablation_config
from train_ablation import completed_result
import train_ablation as training
from train_4_5seed import TRAINING_FAULT_FILL_MULTIPLIER
from training_runner import load_checkpoint_training_config as strict_load_checkpoint_training_config, checkpoint_content_hashes
from training_runner import training_input_hashes
from training_runner import runtime_acceleration_contract


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_TEST_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test'
CONFIG_ROOT = BASE_TEST_ROOT / 'config'
TEST_ROOT = BASE_TEST_ROOT / 'ablation'
MODEL_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'model'
STATUS_ROOT = MODEL_ROOT / 'ablation_training_status'
LOCK_PATH = TEST_ROOT / 'ABLATION_TESTS_RUNNING.lock'
MODEL_MANIFEST_PATH = TEST_ROOT / 'ablation_model_manifest.json'
TEST_PLAN_PATH = TEST_ROOT / 'TEST_PLAN.json'

ENCODERS = (
    'direct_concat', 'gat', 'graphsage', 'jknet', 'no_self_feature',
)
SEEDS = TRAINING_SEEDS
MAX_WORKERS = 15
TEST_BASE_SATELLITE_COUNT = sum(
    line.startswith('1 ')
    for line in (PROJECT_ROOT / 'Satellite_Data' / 'wallker_tles.txt')
    .read_text(encoding='utf-8').splitlines()
)


ISACR_PLUS = False
BASE_OUTPUT_ROOT = TEST_ROOT


def configure_isacr_plus(enabled):
    global ISACR_PLUS, TEST_ROOT, LOCK_PATH, MODEL_MANIFEST_PATH, TEST_PLAN_PATH
    if ISACR_PLUS == enabled:
        return
    ISACR_PLUS = enabled
    TEST_ROOT = BASE_OUTPUT_ROOT.with_name(BASE_OUTPUT_ROOT.name + '_plus') if enabled else BASE_OUTPUT_ROOT
    LOCK_PATH = TEST_ROOT / 'ABLATION_TESTS_RUNNING.lock'
    MODEL_MANIFEST_PATH = TEST_ROOT / 'ablation_model_manifest.json'
    TEST_PLAN_PATH = TEST_ROOT / 'TEST_PLAN.json'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    os.replace(temporary, path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['status'])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


COMMON_TEST = read_json(CONFIG_ROOT / 'common.json')
COMMON_TEST['environment']['fault_fill_multiplier'] = TRAINING_FAULT_FILL_MULTIPLIER
TEST_BASE_PACKET_FREQUENCY = COMMON_TEST['environment']['packet_frequency']
LOAD_TESTS = read_json(CONFIG_ROOT / 'load_experiments.json')
FAULT_TESTS = read_json(CONFIG_ROOT / 'fault_experiments.json')
LOAD_TESTS.update(edge_p=0.18, edge_q=0.82)
FAULT_TESTS['workload'] = {
    field: next(case for case in LOAD_TESTS['cases'] if case['id'] == 'load_160')[field]
    for field in ('target_load_tasks_per_second', 'poisson_rate_g', 'poisson_rate_s')
}
TRAINING_WARMUP_STEPS = build_ablation_config('no_self_feature', 0)['general']['warmup_steps']
TEST_SEED = int(COMMON_TEST['general']['scenario_seed'])
RANDOM_SEED = int(COMMON_TEST['general']['random_seed'])
DURATION = int(COMMON_TEST['general']['duration'])
EXPECTED_TEST_STEP = DURATION // int(COMMON_TEST['general']['time_stride'])
PYTHON_HASH_SEED = str(COMMON_TEST['python_hash_seed'])
os.environ['PYTHONHASHSEED'] = PYTHON_HASH_SEED

LOAD_SETTINGS = tuple(LOAD_TESTS['cases'])
LOAD_FAULT_PROBABILITY = float(LOAD_TESTS['edge_p'])
FAULT_LOAD_SETTING = dict(FAULT_TESTS['workload'])
FAULT_CASES = tuple(FAULT_TESTS['cases'])
FAULT_PROBABILITIES = tuple(float(item['edge_p']) for item in FAULT_CASES)
FAULT_CASE_BY_PROBABILITY = {
    float(item['edge_p']): item for item in FAULT_CASES
}
LOAD_FAULT_CASE = FAULT_CASE_BY_PROBABILITY[LOAD_FAULT_PROBABILITY]
FAULT_WORKLOAD_CASE = next(
    item for item in LOAD_SETTINGS
    if int(item['target_load_tasks_per_second'])
    == int(FAULT_LOAD_SETTING['target_load_tasks_per_second'])
)
CONFIG_NAMES = (
    'common.json',
    'load_experiments.json',
    'fault_experiments.json',
    f'scenario_seed_{TEST_SEED}/manifest.json',
)
CONFIG_HASHES = {
    name: hashlib.sha256((CONFIG_ROOT / name).read_bytes()).hexdigest()
    for name in CONFIG_NAMES
}
SOURCE_HASHES = {
    name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
    for name in (
        'ablation/aggregation_models.py', 'ablation/training_entry.py',
        'ablation/train_graphsage.py', 'ablation/message_aggregation.py', 'train_ablation.py',
        'train_4_5seed.py', 'train_ISACR_self_elimination.py', 'training_runner.py',
        'SatelliteNetworkSimulator_Computing.py', 'Base_Agents.py', 'test_ablation.py',
        'test_config.py',
    )
}
TEST_CONTRACT_HASH = hashlib.sha256(
    json.dumps({'configuration': CONFIG_HASHES, 'source': SOURCE_HASHES,
                'runtime_acceleration': runtime_acceleration_contract()},
               sort_keys=True).encode('ascii')
).hexdigest()


def validate_fixed_configs():
    if COMMON_TEST.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported common test configuration schema')
    if TEST_SEED not in [int(seed) for seed in COMMON_TEST['scenario_seeds']]:
        raise ValueError('scenario_seed is not listed in common configuration')
    manifest = read_json(CONFIG_ROOT / f'scenario_seed_{TEST_SEED}' / 'manifest.json')
    if int(manifest['scenario_seed']) != TEST_SEED:
        raise ValueError('Scenario manifest seed does not match common configuration')
    if int(manifest['duration']) != DURATION:
        raise ValueError('Scenario manifest duration does not match test duration')
    if COMMON_TEST['stream_contract'].get('mode') != 'fixed_trace_files':
        raise ValueError('Ablation tests must use fixed trace files')
    if EXPECTED_TEST_STEP != 300:
        raise ValueError('The fixed test contract must contain exactly 300 steps')
    if [int(item['target_load_tasks_per_second']) for item in LOAD_SETTINGS] != list(range(100, 161, 10)):
        raise ValueError('Unexpected load experiment points')
    if LOAD_FAULT_PROBABILITY != .18 or int(FAULT_LOAD_SETTING['target_load_tasks_per_second']) != 160:
        raise ValueError('Expected load100-160/fault0.18 and load160/fault0-0.18')
    if [float(item['fault_percent']) for item in FAULT_CASES] != [0, 3, 6, 9, 12, 15, 18]:
        raise ValueError('Unexpected fault experiment points')
    if abs(float(LOAD_TESTS['edge_q']) - (1.0 - LOAD_FAULT_PROBABILITY)) > 1e-12:
        raise ValueError('Load experiment edge_q must equal 1-edge_p')
    for item in FAULT_CASES:
        if abs(float(item['edge_q']) - (1.0 - float(item['edge_p']))) > 1e-12:
            raise ValueError(f"Fault case {item['id']} has inconsistent edge_p/edge_q")
    if list(COMMON_TEST['general'].get('reward_factors', ())) != [1, 0.05, 1, 0.1, 0.25]:
        raise ValueError('Ablation reward factors are not aligned')
    if COMMON_TEST['environment'].get('require_computed_before_downlink') is not False:
        raise ValueError('Ablation direct-downlink target rule is disabled')
    if COMMON_TEST['stream_contract'].get('destination_policy') != (
        'fixed_session_destination_direct_downlink_at_target'
    ):
        raise ValueError('Ablation destination policy is not fixed-target direct downlink')


validate_fixed_configs()


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


SCENARIO_MANIFEST = read_json(CONFIG_ROOT / f'scenario_seed_{TEST_SEED}' / 'manifest.json')


def case_trace_entries(kind, case_id):
    workload_id = case_id if kind == 'load' else FAULT_WORKLOAD_CASE['id']
    fault_id = LOAD_FAULT_CASE['id'] if kind == 'load' else case_id
    return (
        SCENARIO_MANIFEST['workloads'][workload_id],
        SCENARIO_MANIFEST['faults'][fault_id],
    )


def number_tag(value):
    return f'{value:g}'.replace('.', 'p')


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
    match = re.search(r'_train_reward_best_reward(m?\d+p\d+)', path.name)
    if not match:
        return float('-inf')
    return float(match.group(1).replace('m', '-').replace('p', '.'))


def model_directory(encoder, seed):
    return MODEL_ROOT / VARIANT_NAMES[encoder] / f'seed{seed}'


def user_accepted_result(seed):
    path = PROJECT_ROOT / 'diagnostics/directconcat_user_acceptance.json'
    if not path.is_file():
        return None
    record = read_json(path)['models'].get(str(seed))
    if not record or record.get('status') != 'user_accepted':
        return None
    for name, digest in record['files_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'User-accepted artifact changed: {name}')
    return record


def load_checkpoint_training_config(checkpoint):
    preserved_launch = PROJECT_ROOT / 'diagnostics/ablation_lz4_workers5_launch_20261021.json'
    if preserved_launch.is_file():
        record = read_json(preserved_launch)
        metadata_path = Path(str(checkpoint) + '.training.json').resolve()
        files = record.get('preserved_original10_sha256', {})
        if str(metadata_path) in files:
            for name, digest in record['source_sha256'].items():
                if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Launch source changed: {name}')
            for name, digest in files.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Preserved model artifact changed: {name}')
            metadata = read_json(metadata_path)
            if metadata['checkpoint_sha256'] != checkpoint_content_hashes(checkpoint):
                raise ValueError('Preserved checkpoint hash mismatch')
            return metadata['config']
    try:
        return strict_load_checkpoint_training_config(checkpoint)
    except ValueError:
        launch = PROJECT_ROOT / 'diagnostics/ablation_native_cuda_launch.json'
        if launch.is_file():
            record = read_json(launch)
            metadata_path = Path(str(checkpoint) + '.training.json')
            files = record.get('preserved_original10_sha256', {})
            # Reuse only exact archived-run artifacts, never relabel metadata.
            if str(metadata_path.resolve()) in files:
                metadata = read_json(metadata_path)
                for path in (metadata_path, *(
                        Path(checkpoint) / name for name in metadata['checkpoint_sha256'])):
                    if path.is_file() and files.get(str(path.resolve())) != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError(f'Preserved checkpoint changed: {path}')
                if metadata['checkpoint_sha256'] != checkpoint_content_hashes(checkpoint):
                    raise ValueError('Preserved checkpoint hash mismatch')
                config = metadata['config']
                current, recorded = training_input_hashes(config), config['training_input_sha256']
                for name in current.keys() | recorded.keys():
                    if current.get(name) == recorded.get(name):
                        continue
                    normalized = Path(name).as_posix()
                    if normalized in ('ablation/train_graphsage.py', 'ablation/message_aggregation.py'):
                        if current.get(name) == record['source_sha256'].get(normalized):
                            continue
                    if normalized == 'training_runner.py':
                        prefix = (PROJECT_ROOT / normalized).read_bytes().split(
                            b'# BEGIN OPTIONAL_RUNTIME_' + b'ACCELERATION_V1\n', 1)[0]
                        if (recorded.get(name) == hashlib.sha256(prefix).hexdigest()
                                and current.get(name) == record['source_sha256'][normalized]):
                            continue
                    raise ValueError(f'Unapproved preserved-model input change: {name}')
                return config
        # The user accepted these exact old-source DirectConcat checkpoints.
        for seed in SEEDS:
            record = user_accepted_result(seed)
            if record and str(checkpoint.resolve()) in (record['reward_checkpoint'], record['latest_checkpoint']):
                metadata = read_json(Path(str(checkpoint) + '.training.json'))
                if metadata['checkpoint_sha256'] != checkpoint_content_hashes(checkpoint):
                    raise ValueError('User-accepted checkpoint hash mismatch')
                config = metadata['config']
                current, recorded = training_input_hashes(config), config['training_input_sha256']
                validation = read_json(PROJECT_ROOT / 'diagnostics/ablation_device_fix_20261020/validation.json')
                for name in current.keys() | recorded.keys():
                    if current.get(name) == recorded.get(name):
                        continue
                    if (Path(name).as_posix() != 'ablation/train_graphsage.py'
                            or recorded.get(name) != validation['before_source_sha256']
                            or current.get(name) != validation['after_source_sha256']):
                        raise ValueError(f'Unapproved source/resource change: {name}')
                return config
        raise


def select_model(encoder, seed):
    result = completed_result(encoder, seed)
    preserved_launch = PROJECT_ROOT / 'diagnostics/ablation_lz4_workers5_launch_20261021.json'
    if result is None and encoder in ('direct_concat', 'no_self_feature') and preserved_launch.is_file():
        record = read_json(preserved_launch)
        config, directory, log = training.paths_for(encoder, seed)
        protected_result = str((directory / 'process_result.json').resolve())
        if protected_result not in record.get('preserved_original10_sha256', {}):
            raise ValueError('Missing preserved result provenance')
        saved = read_json(Path(protected_result))
        accepted = user_accepted_result(seed) if encoder == 'direct_concat' else None
        reward, latest = Path(saved['reward_checkpoint']), Path(saved['latest_checkpoint'])
        if (saved.get('encoder') == encoder and saved.get('seed') == seed
                and saved.get('algorithm') == VARIANT_NAMES[encoder]
                and (saved.get('exit_code') == 0 or accepted)
                and saved.get('final_step') == training.last_step(log) == 1000
                and read_json(directory / 'training_config.json') == _json_ready(config)
                and all(load_checkpoint_training_config(path)[section] == _json_ready(config)[section]
                        for path in (reward, latest) for section in ('general', 'agent', 'environment'))):
            result = saved
    launch_path = PROJECT_ROOT / 'diagnostics/ablation_native_cuda_launch.json'
    if result is None and encoder in ('direct_concat', 'no_self_feature') and launch_path.is_file():
        record = read_json(launch_path)
        if training.IMPLEMENTATION_SHA256 != record['implementation_sha256']:
            raise ValueError('Native training implementation changed since launch')
        for name, digest in record['source_sha256'].items():
            if hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Native training source changed: {name}')
        config, directory, log = training.paths_for(encoder, seed)
        files = {name: digest for name, digest in record['preserved_original10_sha256'].items()
                 if Path(name).is_relative_to(directory)}
        if not files:
            raise ValueError('Missing preserved-model provenance')
        for name, digest in files.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Preserved model artifact changed: {name}')
        saved = read_json(directory / 'process_result.json')
        reward = training.reward_checkpoint(encoder, seed)
        latest = training.find_latest_checkpoint(PROJECT_ROOT / config['agent']['model_path'])
        accepted = user_accepted_result(seed) if encoder == 'direct_concat' else None
        if (saved.get('encoder') == encoder and saved.get('seed') == seed
                and saved.get('algorithm') == VARIANT_NAMES[encoder]
                and (saved.get('exit_code') == 0 or accepted)
                and saved.get('final_step') == config['general']['duration'] == training.last_step(log)
                and read_json(directory / 'training_config.json') == _json_ready(config)
                and saved.get('reward_checkpoint') == str(reward)
                and saved.get('latest_checkpoint') == str(latest)
                and all(path is not None and all(
                    load_checkpoint_training_config(path)[section] == _json_ready(config)[section]
                    for section in ('general', 'agent', 'environment')) for path in (reward, latest))):
            result = saved
    if result is None and encoder in ('direct_concat', 'no_self_feature'):
        # These ten models ran before the device-only re-encoding fix. Verify
        # that exact source transition, without rewriting their contracts.
        evidence = PROJECT_ROOT / 'diagnostics' / 'ablation_device_fix_20261020'
        validation = read_json(evidence / 'validation.json')
        names = (
            'train_ablation.py', 'train_4_5seed.py', 'train_ISACR_self_elimination.py',
            'ablation/aggregation_models.py', 'ablation/message_aggregation.py', 'ablation/training_entry.py',
            'ablation/train_graphsage.py', 'SatelliteNetworkSimulator_Computing.py',
            'SatelliteNetworkSimulator_Beta.py', 'Base_Agents.py', 'training_runner.py',
        )
        original_hash = hashlib.sha256(b''.join(
            (evidence / 'train_graphsage.before.py').read_bytes()
            if name == 'ablation/train_graphsage.py' else (PROJECT_ROOT / name).read_bytes()
            for name in names
        )).hexdigest()
        config, directory, log = training.paths_for(encoder, seed)
        saved_result = read_json(directory / 'process_result.json')
        reward = training.reward_checkpoint(encoder, seed)
        latest = training.find_latest_checkpoint(PROJECT_ROOT / config['agent']['model_path'])
        accepted = user_accepted_result(seed) if encoder == 'direct_concat' else None
        def checkpoint_matches(path):
            if not accepted:
                return training.checkpoint_matches(path, config)
            saved = load_checkpoint_training_config(path)
            return all(saved[section] == _json_ready(config)[section]
                       for section in ('general', 'agent', 'environment'))
        if (original_hash == validation['original_implementation_sha256']
                == saved_result.get('implementation_sha256')
                and training.IMPLEMENTATION_SHA256 == validation['fixed_implementation_sha256']
                and saved_result.get('encoder') == encoder
                and saved_result.get('algorithm') == VARIANT_NAMES[encoder]
                and saved_result.get('seed') == seed
                and (saved_result.get('exit_code') == 0 or accepted)
                and saved_result.get('final_step') == config['general']['duration']
                and training.last_step(log) == config['general']['duration']
                and read_json(directory / 'training_config.json') == _json_ready(config)
                and saved_result.get('training_log') == str(log)
                and saved_result.get('reward_checkpoint') == str(reward)
                and saved_result.get('latest_checkpoint') == str(latest)
                and checkpoint_matches(reward)
                and checkpoint_matches(latest)):
            result = saved_result
    if result is None:
        raise RuntimeError(f'{VARIANT_NAMES[encoder]}/seed{seed}: no completed model matching current training contract')
    return Path(result['reward_checkpoint']).resolve(), Path(result['training_log']).resolve()


def build_case_config(encoder, checkpoint, seed, kind, value, raw_path):
    if kind == 'load':
        load_setting = value
        fault_probability = LOAD_FAULT_PROBABILITY
        case_id = value['id']
    else:
        load_setting = FAULT_LOAD_SETTING
        fault_probability = float(value)
        case_id = FAULT_CASE_BY_PROBABILITY[fault_probability]['id']

    workload_trace, fault_trace = case_trace_entries(kind, case_id)
    workload_trace_path = resolve_trace(workload_trace)
    fault_trace_path = resolve_trace(fault_trace)

    # Start from the exact per-encoder training configuration. Only the
    # phase, fixed-test streams, model path, and per-case load/fault settings
    # are changed here; encoder and aggregation settings remain untouched.
    config = copy.deepcopy(build_ablation_config(encoder, seed))
    trained = load_checkpoint_training_config(checkpoint)
    expected = _json_ready(config)
    if any(trained[section] != expected[section] for section in ('general', 'agent', 'environment')):
        raise ValueError(f'{VARIANT_NAMES[encoder]}/seed{seed}: checkpoint training configuration mismatch')
    # Keep each trained representation's fault/aggregation contract.
    variant_environment = {
        key: config['environment'][key]
        for key in (
            'fault_compute_double_mean',
            'fault_fill_multiplier',
            'historical_self_feature_elimination',
            'disable_self_feature_elimination',
            'gnn_information_hops',
            'gnn_include_self',
            'gnn_node_feature_dim',
        )
        if key in config['environment']
    }
    config['general'].update(COMMON_TEST['general'])
    config['general'].update(
        phase='test',
        random_seed=RANDOM_SEED,
        scenario_seed=TEST_SEED,
        epsilon=0.0,
        print_cycle=1,
        show_test_summary=True,
        model_saving_enabled=False,
    )
    config['agent']['model_path'] = str(checkpoint)
    config['environment'].update(COMMON_TEST['environment'])
    config['environment'].update(variant_environment)
    config['environment'].update(load_setting)
    config['environment']['id'] = load_setting.get('id', FAULT_WORKLOAD_CASE['id'])
    config['environment'].update(
        poisson_rate_s=load_setting['poisson_rate_s'],
        poisson_rate_g=load_setting['poisson_rate_g'],
        packet_frequency=TEST_BASE_PACKET_FREQUENCY,
        edge_p=fault_probability,
        edge_q=1.0 - fault_probability,
        node_p=0.0,
        node_q=0.0,
        markov_mode=True,
        require_computed_before_downlink=False,
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
    for section in ('general', 'environment'):
        for key, expected_value in COMMON_TEST[section].items():
            if config[section].get(key) != expected_value:
                raise ValueError(f'Common test setting changed: {section}.{key}')
    config['test_contract'] = {
        'runtime_acceleration': runtime_acceleration_contract(),
        'algorithm': VARIANT_NAMES[encoder],
        'encoder': encoder,
        'training_seed': seed,
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': checkpoint_content_hashes(checkpoint),
        'case_id': case_id,
        'contract_sha256': TEST_CONTRACT_HASH,
        'configuration_sha256': CONFIG_HASHES,
        'source_sha256': SOURCE_HASHES,
        'python_hash_seed': PYTHON_HASH_SEED,
        'stream_contract': COMMON_TEST['stream_contract'],
        'workload_trace': workload_trace,
        'fault_trace': fault_trace,
    }
    config['test_contract']['isacr_plus'] = ISACR_PLUS
    if ISACR_PLUS:
        config['general']['select_mode'] = 3
        config['test_contract']['destination_selection'] = {
            'method': 'masked_joint_destination_action_q',
            'model_algorithm': VARIANT_NAMES[encoder],
            'source_sha256': SOURCE_HASHES['test_config.py'],
        }
    result_path = model_directory(encoder, seed) / 'process_result.json'
    if result_path.is_file():
        config['test_contract']['training_implementation_sha256'] = read_json(result_path).get('implementation_sha256')
    if encoder == 'direct_concat' and user_accepted_result(seed):
        config['test_contract']['training_acceptance'] = 'user_accepted'
        config['test_contract']['training_acceptance_record'] = str(PROJECT_ROOT / 'diagnostics/directconcat_user_acceptance.json')
    return config


def parse_optional_number(text, label):
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    match = re.search(rf'{re.escape(label)}: ({number}|None)', text)
    return None if not match or match.group(1) == 'None' else float(match.group(1))


def parse_optional_text(text, label):
    match = re.search(rf'{re.escape(label)}: ([^\r\n]+)', text)
    return match.group(1).strip() if match else None


def parse_test_result(raw_path, encoder, seed, kind, load_setting, fault_probability):
    content = raw_path.read_text(encoding='utf-8')
    snapshots = []
    rewards = []
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

    keys = set().union(*(snapshot.keys() for snapshot in snapshots))
    totals = {key: sum(snapshot.get(key, 0) for snapshot in snapshots) for key in keys}
    reached = totals.get('Reached_0', 0) + totals.get('Reached_1', 0)
    lost = (
        totals.get('Lost_upload', 0)
        + totals.get('Lost_relay_0', 0)
        + totals.get('Lost_relay_1', 0)
    )
    finished = reached + lost

    def divide(numerator, denominator):
        return numerator / denominator if denominator else None

    components = {
        name: divide(
            totals.get(f'Total_{name}_delay_0', 0)
            + totals.get(f'Total_{name}_delay_1', 0),
            reached,
        )
        for name in ('computation', 'queueing', 'propagation', 'transmission')
    }
    average_delay = divide(
        totals.get('Total_delay_0', 0) + totals.get('Total_delay_1', 0),
        reached,
    )
    component_sum = (
        sum(components.values())
        if reached and all(value is not None for value in components.values())
        else None
    )
    workload_trace, fault_trace = case_trace_entries(
        kind,
        load_setting['id'] if kind == 'load' else FAULT_CASE_BY_PROBABILITY[fault_probability]['id'],
    )
    row = {
        'algorithm': VARIANT_NAMES[encoder],
        'encoder': encoder,
        'training_seed': seed,
        'scenario_seed': TEST_SEED,
        'isacr_plus': ISACR_PLUS,
        'test_contract_sha256': TEST_CONTRACT_HASH,
        'test_type': kind,
        'test_case_id': load_setting['id'] if kind == 'load' else FAULT_CASE_BY_PROBABILITY[fault_probability]['id'],
        'target_load_tasks_per_second': load_setting['target_load_tasks_per_second'],
        'poisson_rate_s': load_setting['poisson_rate_s'],
        'poisson_rate_g': load_setting['poisson_rate_g'],
        'packet_frequency': TEST_BASE_PACKET_FREQUENCY,
        'edge_p': fault_probability,
        'edge_q': 1.0 - fault_probability,
        'fault_percent': fault_probability * 100,
        'generated_tasks': totals.get('Total', 0),
        'actual_tasks_per_second': divide(totals.get('Total', 0), DURATION),
        'reached_tasks': reached,
        'lost_tasks': lost,
        'unfinished_tasks': totals.get('Total', 0) - finished,
        'finished_tasks_percent': divide(finished * 100, totals.get('Total', 0)),
        'lost_generated_percent': divide(lost * 100, totals.get('Total', 0)),
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
            totals.get('Total_hops_0', 0) + totals.get('Total_hops_1', 0),
            reached,
        ),
        'average_computing_percent': divide(
            totals.get('Is_computing', 0) * 100,
            TEST_BASE_SATELLITE_COUNT * DURATION,
        ),
        'average_waiting_seconds': divide(
            totals.get('Computing_waiting_time', 0), reached,
        ),
        'average_ending_reward': divide(sum(rewards), len(rewards)),
        'workload_trace_count': parse_optional_number(content, 'Workload trace count'),
        'workload_trace_sha256': parse_optional_text(content, 'Workload trace SHA256'),
        'fault_trace_count': parse_optional_number(content, 'Fault trace count'),
        'fault_trace_sha256': parse_optional_text(content, 'Fault trace SHA256'),
        'expected_workload_trace_sha256': workload_trace['sha256'],
        'expected_fault_trace_sha256': fault_trace['sha256'],
        'raw_result_file': str(raw_path),
    }
    if (row['workload_trace_sha256'] != workload_trace['sha256']
            or row['fault_trace_sha256'] != fault_trace['sha256']
            or row['workload_trace_count'] != workload_trace['task_count']
            or row['fault_trace_count'] != fault_trace['event_count']):
        raise RuntimeError(f'Actual evaluation streams do not match the fixed traces: {raw_path}')
    return row


def unit_directory(encoder, seed):
    return TEST_ROOT / VARIANT_NAMES[encoder] / f'seed{seed}'


def case_directory(encoder, seed, kind, value):
    name = (
        f'load_{value["target_load_tasks_per_second"]}'
        if kind == 'load' else f'fault_p{number_tag(value)}'
    )
    return unit_directory(encoder, seed) / kind / name


def complete_result(raw_path):
    return raw_path.is_file() and last_step(raw_path) == EXPECTED_TEST_STEP


def stored_contract_matches(config_path, encoder, seed, kind, value, checkpoint):
    if not config_path.is_file():
        return False
    stored = read_json(config_path)
    contract = stored.get('test_contract', {})
    expected_case = (
        value['id'] if kind == 'load'
        else FAULT_CASE_BY_PROBABILITY[float(value)]['id']
    )
    return (
        contract.get('isacr_plus', False) == ISACR_PLUS
        and (not ISACR_PLUS or contract.get('destination_selection') == {
            'method': 'masked_joint_destination_action_q',
            'model_algorithm': VARIANT_NAMES[encoder],
            'source_sha256': SOURCE_HASHES['test_config.py'],
        })
        and stored['general'].get('select_mode') == (3 if ISACR_PLUS else 1)
        and contract.get('contract_sha256') == TEST_CONTRACT_HASH
        and contract.get('encoder') == encoder
        and int(contract.get('training_seed', -1)) == seed
        and contract.get('case_id') == expected_case
        and contract.get('checkpoint') == str(checkpoint.resolve())
        and contract.get('checkpoint_sha256') == checkpoint_content_hashes(checkpoint)
    )


def run_case(encoder, seed, checkpoint, kind, value, force_rerun=False):
    directory = case_directory(encoder, seed, kind, value)
    raw_path = directory / 'result.txt'
    running_path = directory / 'result.running.txt'
    config_path = directory / 'config.json'
    summary_path = directory / 'summary.json'
    console_path = directory / 'console.txt'
    directory.mkdir(parents=True, exist_ok=True)

    if kind == 'fault' and float(value) == LOAD_FAULT_PROBABILITY:
        # The load/fault intersection is one simulation, represented in both scans.
        shared = run_case(encoder, seed, checkpoint, 'load', FAULT_WORKLOAD_CASE, force_rerun)
        source = Path(shared['raw_result_file'])
        source_config = read_json(source.parent / 'config.json')
        config = build_case_config(encoder, checkpoint, seed, kind, value, raw_path)
        source_environment = dict(source_config['environment'])
        target_environment = dict(config['environment'])
        for environment in (source_environment, target_environment):
            environment.pop('id', None)
            environment.pop('save_training_data', None)
        if (source_environment != target_environment
                or source_config['general'] != config['general']
                or source_config['agent'] != _json_ready(config['agent'])):
            raise RuntimeError('Load/fault intersection simulation settings differ')
        previous = [p for p in (raw_path, running_path, config_path, summary_path, console_path) if p.exists()]
        if previous:
            archive = directory / 'previous_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            archive.mkdir(parents=True)
            for path in previous:
                path.rename(archive / path.name)
        shutil.copy2(source, raw_path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Reused intersection raw result SHA256 mismatch')
        config['test_contract']['reused_result'] = {'source': str(source), 'sha256': digest}
        write_json(config_path, config)
        row = parse_test_result(raw_path, encoder, seed, kind, FAULT_WORKLOAD_CASE, float(value))
        row['model_path'] = str(checkpoint)
        row['model_sha256'] = json.dumps(checkpoint_content_hashes(checkpoint), sort_keys=True)
        row['fault_fill_multiplier'] = config['environment']['fault_fill_multiplier']
        write_json(summary_path, row)
        return row

    if not force_rerun and complete_result(raw_path):
        if not stored_contract_matches(config_path, encoder, seed, kind, value, checkpoint):
            raise RuntimeError(f'Existing result has a mismatched test contract: {raw_path}')
        load_setting = value if kind == 'load' else FAULT_WORKLOAD_CASE
        probability = LOAD_FAULT_PROBABILITY if kind == 'load' else float(value)
        row = parse_test_result(raw_path, encoder, seed, kind, load_setting, probability)
        row['model_path'] = str(checkpoint)
        row['model_sha256'] = json.dumps(checkpoint_content_hashes(checkpoint), sort_keys=True)
        row['fault_fill_multiplier'] = read_json(config_path)['environment']['fault_fill_multiplier']
        write_json(summary_path, row)
        return row

    previous = [p for p in (raw_path, running_path, config_path, summary_path, console_path) if p.exists()]
    if previous:
        archive = directory / 'previous_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        archive.mkdir(parents=True)
        for path in previous:
            path.rename(archive / path.name)
    config = build_case_config(encoder, checkpoint, seed, kind, value, running_path)
    prepare_test_acceleration(config)
    write_json(config_path, config)
    with console_path.open('w', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            with destination_selection(ablation=True) if ISACR_PLUS else nullcontext():
                run_training(config)
    if not complete_result(running_path):
        raise RuntimeError(
            f'Incomplete result: {running_path}; expected final step '
            f'{EXPECTED_TEST_STEP}, got {last_step(running_path)}'
        )
    running_path.replace(raw_path)
    config['environment']['save_training_data'] = str(raw_path)
    write_json(config_path, config)
    load_setting = (
        value if kind == 'load' else FAULT_WORKLOAD_CASE
    )
    fault_probability = (
        LOAD_FAULT_PROBABILITY if kind == 'load' else float(value)
    )
    row = parse_test_result(
        raw_path, encoder, seed, kind, load_setting, fault_probability,
    )
    row['model_path'] = str(checkpoint)
    row['model_sha256'] = json.dumps(checkpoint_content_hashes(checkpoint), sort_keys=True)
    row['fault_fill_multiplier'] = config['environment']['fault_fill_multiplier']
    write_json(summary_path, row)
    return row


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def save_unit_status(encoder, seed, checkpoint, rows, errors, expected_cases):
    write_json(
        unit_directory(encoder, seed) / 'STATUS.json',
        {
            'algorithm': VARIANT_NAMES[encoder],
            'encoder': encoder,
            'training_seed': seed,
            'model_path': str(checkpoint) if checkpoint else None,
            'expected_cases': expected_cases,
            'completed_cases': len(rows),
            'missing_cases': expected_cases - len(rows),
            'errors': errors,
            'status': 'complete' if len(rows) == expected_cases and not errors else 'partial',
            'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        },
    )


def run_unit(job):
    encoder, seed, load_settings, fault_probabilities, force_rerun = job
    jobs = [('load', item) for item in load_settings] + [
        ('fault', value) for value in fault_probabilities
    ]
    selected_ids = {value['id'] if kind == 'load' else FAULT_CASE_BY_PROBABILITY[float(value)]['id']
                    for kind, value in jobs}
    unit_dir = unit_directory(encoder, seed)
    unit_dir.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint, training_log = select_model(encoder, seed)
    except Exception as error:
        save_unit_status(encoder, seed, None, [], [str(error)], len(jobs))
        return {
            'algorithm': VARIANT_NAMES[encoder],
            'encoder': encoder,
            'training_seed': seed,
            'status': 'not_ready',
            'reason': str(error),
        }

    load_csv = unit_dir / 'load' / 'load_test_results.csv'
    fault_csv = unit_dir / 'fault' / 'fault_test_results.csv'
    load_rows = current_result_rows(read_rows(load_csv), encoder, seed)
    fault_rows = current_result_rows(read_rows(fault_csv), encoder, seed)
    errors = []
    # Remove selected cached rows until each case has been revalidated.
    load_rows = [row for row in load_rows if row['test_case_id'] not in selected_ids]
    fault_rows = [row for row in fault_rows if row['test_case_id'] not in selected_ids]
    for kind, value in jobs:
        try:
            reuse_intersection = (
                kind == 'fault' and float(value) == LOAD_FAULT_PROBABILITY
                and any(item['id'] == FAULT_WORKLOAD_CASE['id'] for item in load_settings)
            )
            row = run_case(encoder, seed, checkpoint, kind, value,
                           force_rerun and not reuse_intersection)
            target = load_rows if kind == 'load' else fault_rows
            key = (
                int(float(row['target_load_tasks_per_second']))
                if kind == 'load' else float(row['fault_percent'])
            )
            target[:] = [
                item for item in target
                if (
                    int(float(item['target_load_tasks_per_second']))
                    if kind == 'load' else float(item['fault_percent'])
                ) != key
            ]
            target.append(row)
            write_rows(load_csv, load_rows)
            write_rows(fault_csv, fault_rows)
            save_unit_status(
                encoder, seed, checkpoint,
                [row for row in load_rows + fault_rows if row['test_case_id'] in selected_ids], errors, len(jobs),
            )
        except Exception as error:
            errors.append(f'{kind}:{value}: {error}')
            save_unit_status(
                encoder, seed, checkpoint,
                [row for row in load_rows + fault_rows if row['test_case_id'] in selected_ids], errors, len(jobs),
            )

    selected_rows = [row for row in load_rows + fault_rows if row['test_case_id'] in selected_ids]
    completed = len(selected_rows)
    write_rows(load_csv, load_rows)
    write_rows(fault_csv, fault_rows)
    save_unit_status(encoder, seed, checkpoint, selected_rows, errors, len(jobs))
    return {
        'algorithm': VARIANT_NAMES[encoder],
        'encoder': encoder,
        'training_seed': seed,
        'status': 'complete' if completed == len(jobs) and not errors else 'partial',
        'completed_cases': completed,
        'expected_cases': len(jobs),
        'errors': errors,
        'training_log': str(training_log),
        'model_path': str(checkpoint),
    }


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
    for key, items in sorted(groups.items()):
        result = dict(zip(group_fields, key))
        result['seed_count'] = len({item.get('training_seed') for item in items})
        fields = set().union(*(item.keys() for item in items))
        for field in sorted(fields):
            if field in group_fields or field in {
                'training_seed', 'scenario_seed', 'model_path',
                'raw_result_file', 'test_type', 'encoder',
            }:
                continue
            values = [numeric(item.get(field)) for item in items]
            values = [value for value in values if value is not None]
            if values:
                result[field] = sum(values) / len(values)
        output.append(result)
    return output


def fairness_audit(rows, expected_units=None, expected_case_ids=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get('test_case_id', '')].append(row)
    cases = []
    expected_units = expected_units if expected_units is not None else {(e, s) for e in ENCODERS for s in SEEDS}
    expected_results = len(expected_units)
    for case_id, items in sorted(groups.items()):
        workload_hashes = {item.get('expected_workload_trace_sha256') for item in items}
        fault_hashes = {item.get('expected_fault_trace_sha256') for item in items}
        contracts = {item.get('test_contract_sha256') for item in items}
        actual_workload_hashes = {item.get('workload_trace_sha256') for item in items}
        actual_fault_hashes = {item.get('fault_trace_sha256') for item in items}
        verified = (
            len(items) == expected_results
            and {(item.get('encoder'), int(item.get('training_seed', -1))) for item in items}
                == expected_units
            and len(workload_hashes) == 1
            and len(fault_hashes) == 1
            and actual_workload_hashes == workload_hashes
            and actual_fault_hashes == fault_hashes
            and contracts == {TEST_CONTRACT_HASH}
        )
        cases.append({
            'case_id': case_id,
            'result_count': len(items),
            'status': 'verified' if verified else 'unverified_or_mismatch',
        })
    if expected_case_ids is None:
        expected_case_ids = {item['id'] for item in LOAD_SETTINGS + FAULT_CASES}
    return {
        'status': (
            'verified'
            if {item['case_id'] for item in cases} == expected_case_ids
            and all(item['status'] == 'verified' for item in cases)
            else 'unverified_or_mismatch'
        ),
        'cases': cases,
    }


def current_result_rows(rows, encoder, seed):
    models, current = {}, {}
    for row in rows:
        if (str(row.get('isacr_plus', False)).lower() != str(ISACR_PLUS).lower()
                or row.get('test_contract_sha256') != TEST_CONTRACT_HASH
                or row.get('encoder') != encoder or str(row.get('training_seed')) != str(seed)):
            continue
        checkpoint = row.get('model_path')
        if not checkpoint or not Path(checkpoint).exists():
            continue
        if checkpoint not in models:
            models[checkpoint] = json.dumps(checkpoint_content_hashes(Path(checkpoint)), sort_keys=True)
        if row.get('model_sha256') == models[checkpoint]:
            current[(row['test_type'], row['test_case_id'])] = row
    return list(current.values())


def save_global_summaries(statuses, selection=None):
    load_rows = []
    fault_rows = []
    for encoder in ENCODERS:
        for seed in SEEDS:
            unit_dir = unit_directory(encoder, seed)
            load_rows.extend(current_result_rows(read_rows(unit_dir / 'load' / 'load_test_results.csv'), encoder, seed))
            fault_rows.extend(current_result_rows(read_rows(unit_dir / 'fault' / 'fault_test_results.csv'), encoder, seed))
    algorithms, seeds, loads, faults = selection or (ENCODERS, SEEDS, LOAD_SETTINGS, FAULT_PROBABILITIES)
    expected_units = {(e, s) for e in algorithms for s in seeds}
    load_ids = {item['id'] for item in loads}
    fault_ids = {FAULT_CASE_BY_PROBABILITY[p]['id'] for p in faults}
    load_rows = [r for r in load_rows if (r['encoder'], int(r['training_seed'])) in expected_units
                 and r['test_case_id'] in load_ids]
    fault_rows = [r for r in fault_rows if (r['encoder'], int(r['training_seed'])) in expected_units
                  and r['test_case_id'] in fault_ids]
    write_rows(TEST_ROOT / 'all_load_results.csv', load_rows)
    write_rows(TEST_ROOT / 'all_fault_results.csv', fault_rows)
    write_rows(
        TEST_ROOT / 'ablation_load_mean.csv',
        mean_rows(load_rows, ('algorithm', 'target_load_tasks_per_second')),
    )
    write_rows(
        TEST_ROOT / 'ablation_fault_mean.csv',
        mean_rows(fault_rows, ('algorithm', 'fault_percent')),
    )
    all_rows = load_rows + fault_rows
    audit = fairness_audit(all_rows, expected_units, load_ids | fault_ids)
    expected_cases = len(expected_units) * (len(load_ids) + len(fault_ids))
    completed_cases = len(all_rows)
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
            'algorithms': [VARIANT_NAMES[item] for item in algorithms],
            'encoders': list(algorithms),
            'training_seeds': list(seeds),
            'load_points': [item['target_load_tasks_per_second'] for item in loads],
            'load_fault_probability': LOAD_FAULT_PROBABILITY,
            'fault_load': FAULT_LOAD_SETTING,
            'fault_percentages': [p * 100 for p in faults],
            'duration_seconds': DURATION,
            'test_scenario_seed': TEST_SEED,
            'test_random_seed': RANDOM_SEED,
            'python_hash_seed': PYTHON_HASH_SEED,
            'isacr_plus': ISACR_PLUS,
            'test_contract_sha256': TEST_CONTRACT_HASH,
            'configuration_sha256': CONFIG_HASHES,
            'fairness_audit': audit,
            'max_workers': MAX_WORKERS,
            'statuses': statuses,
            'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        },
    )


def live_training_manager():
    native_path = PROJECT_ROOT / 'diagnostics/ablation_native_cuda_launch.json'
    if native_path.is_file():
        import psutil
        record = read_json(native_path)
        identities = [record, *record.get('process_identities', [])]
        identities.extend(read_json(path) for path in MODEL_ROOT.glob('*/seed*/worker_identity.json'))
        for identity in identities:
            try:
                process = psutil.Process(identity['pid'])
                if abs(process.create_time() - identity['create_time']) < .01:
                    return native_path, process.pid
            except (KeyError, psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    for manifest in ('ablation_original_process_identities.json', 'ablation_device_fix_retry_launch.json'):
        path = PROJECT_ROOT / 'diagnostics' / manifest
        if path.is_file():
            import psutil
            record = read_json(path)
            identities = record.get('processes', []) if isinstance(record, dict) else record
            if manifest == 'ablation_original_process_identities.json':
                # Orphan pool processes may remain idle after writing all results.
                results = [(e, s, model_directory(e, s) / 'process_result.json')
                           for e in ('direct_concat', 'no_self_feature') for s in SEEDS]
                if all(p.is_file() and (read_json(p).get('exit_code') == 0
                       or (e == 'direct_concat' and user_accepted_result(s))) for e, s, p in results):
                    continue
            if manifest == 'ablation_device_fix_retry_launch.json':
                identities = [record]
            for identity in identities:
                if manifest == 'ablation_original_process_identities.json' and not any(
                        'python' in str(arg).lower() for arg in identity.get('cmd', [])):
                    continue
                try:
                    process = psutil.Process(identity['pid'])
                    if abs(process.create_time() - identity['create_time']) < .01:
                        return path, process.pid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    import psutil
    for root in (STATUS_ROOT, MODEL_ROOT / 'ablation_device_retry_status'):
        lock_path = root / 'TRAINING_RUNNING.lock'
        if lock_path.is_file():
            try:
                process = psutil.Process(int(lock_path.read_text(encoding='ascii').strip()))
                if any(Path(arg).name in ('train_ablation.py', 'train_ablation_device_retry.py')
                       for arg in process.cmdline()):
                    return lock_path, process.pid
            except (OSError, ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return None, None


def acquire_lock():
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, 'w', encoding='ascii') as stream:
        stream.write(f'pid={os.getpid()}\n')
    return True


def validate_ready_models(algorithms, seeds):
    models, errors = [], {}
    for encoder in algorithms:
        for seed in seeds:
            try:
                checkpoint, training_log = select_model(encoder, seed)
                models.append(dict(encoder=encoder, algorithm=VARIANT_NAMES[encoder], training_seed=seed,
                                   checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_content_hashes(checkpoint),
                                   training_implementation_sha256=read_json(model_directory(encoder, seed) / 'process_result.json')['implementation_sha256'],
                                   training_log=str(training_log), training_final_step=last_step(training_log)))
            except Exception as error:
                errors[f'{encoder}/seed{seed}'] = str(error)
    write_json(MODEL_MANIFEST_PATH, dict(expected_models=len(algorithms)*len(seeds),ready_models=len(models),
               reference_warmup_steps=TRAINING_WARMUP_STEPS, ablation_warmup_steps=TRAINING_WARMUP_STEPS,
               algorithms=[VARIANT_NAMES[e] for e in algorithms],training_seeds=list(seeds),models=models,errors=errors))
    return models, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_acceleration_argument(parser)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    algorithm_group = parser.add_mutually_exclusive_group()
    algorithm_group.add_argument('--algorithm', choices=ENCODERS)
    algorithm_group.add_argument('--algorithms', nargs='+', choices=ENCODERS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument(
        '--experiment', choices=('point', 'both', 'load', 'fault'), default='point',
        help='Default: one load160/fault0.18 case per model; both runs full sweeps.',
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
    parser.add_argument('--dry-run', action='store_true', help='Write the test matrix without loading models or starting simulations.')
    parser.add_argument('--rerun', action='store_true', help='Archive and rerun selected cases even when complete results exist.')
    parser.add_argument('--isacr-plus', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable joint destination/action selection using existing weights.')
    args = parser.parse_args()
    configure_acceleration(args.acceleration == 'on')
    configure_isacr_plus(args.isacr_plus)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f'--workers must be between 1 and {MAX_WORKERS}')
    if args.experiment == 'load' and args.fault_percent is not None:
        parser.error('--fault-percent cannot be used with --experiment load')
    if args.experiment == 'fault' and args.load_point is not None:
        parser.error('--load-point cannot be used with --experiment fault')
    if args.experiment == 'point':
        if args.load_point not in (None, 160) or args.fault_percent not in (None, 18):
            parser.error('--experiment point is fixed at load160/fault18%; use load/fault/both for sweeps')
        args.load_point = 160

    algorithms = tuple(args.algorithms) if args.algorithms else ((args.algorithm,) if args.algorithm else ENCODERS)
    if len(set(algorithms)) != len(algorithms):
        parser.error('--algorithms must not contain duplicates')
    seeds = (args.seed,) if args.seed is not None else SEEDS
    selected_load_settings = tuple(
        item for item in reversed(LOAD_SETTINGS)
        if args.experiment != 'fault'
        and (args.load_point is None
             or int(item['target_load_tasks_per_second']) == args.load_point)
    )
    selected_fault_probabilities = tuple(
        probability for probability in reversed(FAULT_PROBABILITIES)
        if args.experiment not in ('load', 'point')
        and (args.fault_percent is None
             or int(round(probability * 100)) == args.fault_percent)
    )
    jobs = [
        (encoder, seed, selected_load_settings, selected_fault_probabilities, args.rerun)
        for encoder in algorithms for seed in seeds
    ]
    selection = (algorithms, seeds, selected_load_settings, selected_fault_probabilities)
    if args.dry_run:
        cases = []
        for kind, values in (('load', selected_load_settings), ('fault', selected_fault_probabilities)):
            for value in values:
                case_id = value['id'] if kind == 'load' else FAULT_CASE_BY_PROBABILITY[float(value)]['id']
                workload, fault = case_trace_entries(kind, case_id)
                cases.append(dict(test_type=kind, case_id=case_id,
                    workload=value if kind == 'load' else FAULT_LOAD_SETTING,
                    edge_p=LOAD_FAULT_PROBABILITY if kind == 'load' else float(value),
                    workload_trace=workload, fault_trace=fault))
        write_json(TEST_PLAN_PATH, dict(algorithms=[VARIANT_NAMES[e] for e in algorithms],
            encoders=algorithms, training_seeds=seeds, workers=min(args.workers, len(jobs)),
            expected_models=len(jobs), cases_per_model=len(cases), expected_cases=len(jobs)*len(cases),
            duration_seconds=DURATION, scenario_seed=TEST_SEED, random_seed=RANDOM_SEED,
            configuration_sha256=CONFIG_HASHES, cases=cases, isacr_plus=ISACR_PLUS,
            acceleration=args.acceleration,
            destination_selection='masked_joint_destination_action_q' if ISACR_PLUS else None,
            output_root=str(TEST_ROOT), reference_warmup_steps=TRAINING_WARMUP_STEPS,
            ablation_warmup_steps=TRAINING_WARMUP_STEPS))
        print(f'DRY_RUN: {len(jobs)} models, {len(jobs)*len(cases)} cases; {TEST_PLAN_PATH}', flush=True)
        return
    if args.status_only:
        save_global_summaries([], selection)
        return

    training_lock, training_pid = live_training_manager()
    if training_pid is not None:
        parser.error(
            f'ablation training is still running: pid={training_pid}, lock={training_lock}'
        )
    if not acquire_lock():
        print(f'TEST_STATUS=busy: {LOCK_PATH}', flush=True)
        return

    statuses = []
    try:
        ready_models, model_errors = validate_ready_models(algorithms, seeds)
        if model_errors:
            parser.error(f'model validation failed: {model_errors}')
        if len(ready_models) != len(algorithms) * len(seeds):
            parser.error(f'model validation failed: {len(ready_models)}/{len(algorithms)*len(seeds)} ready')
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                     initializer=configure_isacr_plus, initargs=(args.isacr_plus,)) as executor:
            futures = {executor.submit(run_unit, job): job for job in jobs}
            for future in as_completed(futures):
                status = future.result()
                statuses.append(status)
                print(json.dumps(status, ensure_ascii=False), flush=True)
        save_global_summaries(statuses, selection)
        expected = len(algorithms) * len(seeds) * (
            len(selected_load_settings) + len(selected_fault_probabilities)
        )
        completed = sum(
            int(status.get('completed_cases', 0)) for status in statuses
        )
        print(f'COMPLETE_CASES={completed}/{expected}', flush=True)
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
