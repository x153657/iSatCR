"""Compare 2-hop/4-hop ISACR+, baseline ISACR+ and D3QN+ on four constellations.

Fixed load 160, edge_p=.18, fill multiplier=2.0, training seeds 0-4.
Reuse the five-coefficient runner's fixed constellation traces and result parser.
Baseline ISACR+ on all four constellations and Walker D3QN+ reuse permissive
results. All other cases require computation before downlink.
"""
import argparse
from test_config import (
    add_acceleration_argument, configure_acceleration, prepare_test_acceleration, run_training,
)
import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
import psutil
import SatelliteNetworkSimulator_Beta as satellite_beta

import test_5ix_5seed_5sat as reference
import train_4_5seed as baseline
import train_isacr_2_4_5seed as hop_training
from training_runner import load_checkpoint_training_config, archive_test_outputs
from training_runner import runtime_acceleration_contract


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / 'training_process_data/v1/test/test/isacr_2_4_d3qn_4sat_5seed'
LOCK_PATH = OUTPUT_ROOT / 'TESTS_RUNNING.lock'
ALGORITHMS = ('ISACR_2hop', 'ISACR_4hop', 'ISACR_self_elimination', 'D3QN+')
HOPS = {'ISACR_2hop': 2, 'ISACR_4hop': 4}
SEEDS = baseline.SEEDS
CONSTELLATIONS = reference.ALL_CONSTELLATIONS
SOURCE_HASHES = {name: reference.digest(PROJECT_ROOT / name) for name in (
    'test_isacr_2_4_d3qn_4sat_5seed.py', 'train_isacr_2_4_5seed.py',
    'train_4_5seed.py', 'train_ISACR_self_elimination.py', 'train_D3QN.py',
    *reference.SIMULATION_SOURCE_HASHES,
)}
# The predecessor differs only in status I/O and equivalent message dispatch.
PREDECESSOR_SOURCE_SHA256 = '21fdc463d42c8c720358159920d75c834ebd113fcba9d418f4fbf7acaf1ccb34'
PYTHON_MESSAGE_GENERATORS = {
    name: getattr(satellite_beta.Propagator, name)
    for name in ('send_state', 'send_adjacency_table')
}
read_json = reference.read_json


def write_json(path, value):
    """Publish atomically, tolerating brief Windows reader/antivirus locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                     dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        temporary = Path(stream.name)
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1)


def matching_contract(recorded, expected):
    if recorded == expected:
        return True
    predecessor = copy.deepcopy(expected)
    predecessor['source_sha256'][Path(__file__).name] = PREDECESSOR_SOURCE_SHA256
    return recorded == predecessor


def training_config(algorithm, seed):
    if algorithm in HOPS:
        return hop_training.build_training_config(HOPS[algorithm], seed)
    return baseline.build_training_config('D3QN' if algorithm == 'D3QN+' else algorithm, seed)


def validate_checkpoint(algorithm, seed, checkpoint):
    loader = (reference.load_checkpoint_training_config
              if algorithm == 'ISACR_self_elimination' else load_checkpoint_training_config)
    config = loader(checkpoint)
    expected = training_config(algorithm, seed)
    ignored = {'general': {'algorithm_name'}, 'agent': {'model_path'},
               'environment': {'save_training_data'}}
    names = {'D3QN' if algorithm == 'D3QN+' else algorithm}
    if algorithm == 'ISACR_self_elimination':
        # Coefficient-scan models may also be stored in the baseline directory.
        names.add('ISACR_fault_self_elimination')
    if config['general']['algorithm_name'] not in names:
        raise ValueError(f'Wrong checkpoint algorithm: {checkpoint}')
    for section, excluded in ignored.items():
        for key in expected[section].keys() | config[section].keys():
            if key not in excluded and json.dumps(config[section].get(key)) != json.dumps(expected[section].get(key)):
                raise ValueError(f'{algorithm}/seed{seed}: checkpoint mismatch: {section}.{key}')
    return config


def select_model(algorithm, seed):
    if algorithm in HOPS:
        completed = hop_training.completed_result(HOPS[algorithm], seed)
        if completed is None:
            raise RuntimeError(f'{algorithm}/seed{seed}: training is not complete or validation failed')
        checkpoint = Path(completed['reward_checkpoint'])
    else:
        directory = baseline.MODEL_ROOT / ('D3QN' if algorithm == 'D3QN+' else algorithm)
        checkpoints = list(directory.glob(f'*_seed{seed}_train_reward_best_*.pth'))
        if len(checkpoints) != 1:
            raise RuntimeError(f'{algorithm}/seed{seed}: expected one reward-best checkpoint, got {len(checkpoints)}')
        checkpoint = checkpoints[0]
        logs = list(directory.glob(f'train_*_seed{seed}_*.txt'))
        if not any(reference.last_step(log) == baseline.TRAINING_DURATION for log in logs):
            raise RuntimeError(f'{algorithm}/seed{seed}: no complete step-1000 training log')
    validate_checkpoint(algorithm, seed, checkpoint)
    return checkpoint


def case_directory(constellation, algorithm, seed):
    return OUTPUT_ROOT / constellation / algorithm / f'seed{seed}'


def reuse_directory(constellation, algorithm, seed):
    if algorithm == 'ISACR_self_elimination':
        return PROJECT_ROOT / 'training_process_data/v1/test/test/fault/self_elimination_plus' / constellation / f'seed{seed}/coefficient_2p0'
    if algorithm == 'D3QN+' and constellation == 'Walker':
        return PROJECT_ROOT / 'training_process_data/v1/test/test/D3QN+' / f'seed{seed}/load/load_160'
    return None


def copy_case(constellation, algorithm, seed, checkpoint):
    source = reuse_directory(constellation, algorithm, seed)
    saved = read_json(source / 'config.json')
    expected = build_case_config(constellation, algorithm, seed, checkpoint)
    if reference.last_step(source / 'result.txt') != reference.EXPECTED_STEP:
        raise ValueError(f'Incomplete reuse source: {source}')
    if algorithm == 'ISACR_self_elimination' and read_json(source / 'process_result.json')['exit_code'] != 0:
        raise ValueError(f'Failed reuse source: {source}')
    for section, excluded in (
        ('general', {'algorithm_name'}), ('agent', {'model_path'}),
        ('environment', {'save_training_data'}),
    ):
        for key in expected[section].keys() | saved[section].keys():
            if key not in excluded and expected[section].get(key) != saved[section].get(key):
                if (section == 'environment' and key in ('workload_trace_path', 'fault_trace_path')
                        and reference.digest(Path(saved[section][key]))
                        == reference.digest(Path(expected[section][key]))):
                    continue  # Walker's two entry points keep identical traces in different directories.
                raise ValueError(f'Reuse source mismatch: {source}: {section}.{key}')
    if reference.digest(Path(saved['agent']['model_path'])) != reference.digest(checkpoint):
        raise ValueError(f'Reuse checkpoint mismatch: {source}')
    data = reference.CONFIGS[constellation]
    summary = reference.parse_test_result(source / 'result.txt', algorithm, data['load'],
                                          reference.DURATION, satellite_count=data['satellite_count'])
    for field, trace in (('workload_trace_sha256', 'workload'), ('fault_trace_sha256', 'fault_trace')):
        if summary[field] != data[trace]['sha256']:
            raise ValueError(f'Reuse trace mismatch: {source}: {field}')
    directory = case_directory(constellation, algorithm, seed)
    directory.mkdir(parents=True, exist_ok=True)
    archive_test_outputs([directory / name for name in ('result.txt', 'config.json', 'summary.json', 'process_result.json')])
    shutil.copy2(source / 'result.txt', directory / 'result.txt')
    summary.update(constellation=constellation, training_seed=seed, scenario_seed=reference.TEST_SEED,
                   edge_p=reference.FAULT_PROBABILITY, fault_fill_multiplier=2.0,
                   raw_result_file=str(directory / 'result.txt'), test_contract=expected['test_contract'],
                   copied_from=str(source), source_result_sha256=reference.digest(source / 'result.txt'))
    write_json(directory / 'config.json', expected)
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'process_result.json', {'exit_code': 0, 'copied_from': str(source)})
    return summary


def build_case_config(constellation, algorithm, seed, checkpoint=None):
    config = (validate_checkpoint(algorithm, seed, checkpoint) if checkpoint is not None
              else training_config(algorithm, seed))
    config = copy.deepcopy(config)
    data = reference.CONFIGS[constellation]
    config['general'].update(data['common']['general'])
    config['general'].update(algorithm_name=algorithm, model_saving_enabled=False)
    config['general'].update(random_seed=reference.TEST_SEED, scenario_seed=reference.TEST_SEED,
                             print_cycle=1, show_test_summary=True)
    config['general']['select_mode'] = 3
    config['agent']['model_path'] = str(checkpoint) if checkpoint is not None else None
    config['agent']['update_cycle'] = reference.UPDATE_CYCLE
    config['environment'].update(data['common']['environment'])
    config['environment'].update(data['load'])
    config['environment'].update(
        edge_p=reference.FAULT_PROBABILITY, edge_q=1.0 - reference.FAULT_PROBABILITY,
        node_p=0.0, node_q=0.0, markov_mode=True, fault_compute_double_mean=True,
        fault_fill_multiplier=baseline.TRAINING_FAULT_FILL_MULTIPLIER,
        require_computed_before_downlink=reuse_directory(constellation, algorithm, seed) is None,
        visualize=False, save_log=False,
        save_training_data=str(case_directory(constellation, algorithm, seed) / 'result.running.txt'),
        workload_trace_path=str(data['workload_path']),
        workload_trace_sha256=data['workload']['sha256'],
        workload_trace_task_count=int(data['workload']['task_count']),
        fault_trace_path=str(data['fault_trace_path']),
        fault_trace_sha256=data['fault_trace']['sha256'],
        fault_trace_event_count=int(data['fault_trace']['event_count']),
    )
    contract = {
        'constellation': constellation, 'algorithm': algorithm, 'training_seed': seed,
        'isacr_plus': algorithm != 'D3QN+',
        'd3qn_plus': algorithm == 'D3QN+',
        'require_computed_before_downlink': config['environment']['require_computed_before_downlink'],
        'scenario_seed': reference.TEST_SEED, 'python_hash_seed': reference.PYTHON_HASH_SEED,
        'configuration_sha256': data['hashes'], 'source_sha256': SOURCE_HASHES,
        'runtime_acceleration': runtime_acceleration_contract(),
        'workload_trace_sha256': data['workload']['sha256'],
        'fault_trace_sha256': data['fault_trace']['sha256'],
        'stream_contract': data['common']['stream_contract'],
        'checkpoint': str(Path(checkpoint).resolve()) if checkpoint is not None else None,
        'checkpoint_sha256': reference.digest(checkpoint) if checkpoint is not None else None,
    }
    # Include the complete evaluation configuration in the resumption identity.
    contract['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config['test_contract'] = contract
    return config


def case_complete(constellation, algorithm, seed, checkpoint):
    directory = case_directory(constellation, algorithm, seed)
    try:
        if reference.last_step(directory / 'result.txt') != reference.EXPECTED_STEP:
            return False
        config = read_json(directory / 'config.json')
        summary = read_json(directory / 'summary.json')
        expected = build_case_config(constellation, algorithm, seed, checkpoint)['test_contract']
        return (matching_contract(config['test_contract'], expected)
                and summary.get('test_contract') == config['test_contract']
                and read_json(directory / 'process_result.json').get('exit_code') == 0)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_case(job):
    constellation, algorithm, seed, checkpoint_text, rerun = job
    checkpoint = Path(checkpoint_text)
    directory = case_directory(constellation, algorithm, seed)
    if not rerun and case_complete(constellation, algorithm, seed, checkpoint):
        return read_json(directory / 'summary.json')
    directory.mkdir(parents=True, exist_ok=True)
    archive_test_outputs([directory / name for name in (
        'result.txt', 'result.running.txt', 'config.json', 'summary.json',
        'console.txt', 'process_result.json')])
    config = build_case_config(constellation, algorithm, seed, checkpoint)
    prepare_test_acceleration(config)
    # Keep long-lived SimPy message generators under Python lifetime management.
    # Compiled generators remain a suspect after repeated native access violations.
    for name, function in PYTHON_MESSAGE_GENERATORS.items():
        setattr(satellite_beta.Propagator, name, function)
    config.setdefault('test_acceleration', {})['python_message_generators'] = True
    write_json(directory / 'config.json', config)
    try:
        context = hop_training.isacr_hop_environment(HOPS[algorithm]) if algorithm in HOPS else nullcontext()
        with (directory / 'console.txt').open('w', encoding='utf-8', buffering=1) as console:
            with redirect_stdout(console), redirect_stderr(console), context:
                run_training(config)
        running = directory / 'result.running.txt'
        if reference.last_step(running) != reference.EXPECTED_STEP:
            raise RuntimeError(f'Incomplete test result: {running}')
        data = reference.CONFIGS[constellation]
        summary = reference.parse_test_result(running, algorithm, data['load'], reference.DURATION,
                                              satellite_count=data['satellite_count'])
        for key, trace in (('workload_trace_sha256', 'workload'), ('fault_trace_sha256', 'fault_trace')):
            if summary[key] != data[trace]['sha256']:
                raise RuntimeError(f'Test did not consume the expected fixed trace: {key}')
        running.replace(directory / 'result.txt')
        summary.update(constellation=constellation, training_seed=seed,
                       scenario_seed=reference.TEST_SEED, edge_p=.18, fault_fill_multiplier=baseline.TRAINING_FAULT_FILL_MULTIPLIER,
                       raw_result_file=str(directory / 'result.txt'),
                       test_contract=config['test_contract'])
        write_json(directory / 'summary.json', summary)
        write_json(directory / 'process_result.json', {'exit_code': 0})
        return summary
    except Exception as error:
        write_json(directory / 'process_result.json', {'exit_code': 1, 'error': str(error)})
        raise


def write_aggregates(rows):
    reference.write_rows(OUTPUT_ROOT / 'all_results.csv',
                         [{key: value for key, value in row.items() if key != 'test_contract'} for row in rows])
    means = []
    for constellation in CONSTELLATIONS:
        for algorithm in ALGORITHMS:
            selected = [row for row in rows if row['constellation'] == constellation and row['algorithm'] == algorithm]
            if not selected:
                continue
            mean = {'constellation': constellation, 'algorithm': algorithm,
                    'completed_seeds': len(selected), 'expected_seeds': 5,
                    'all_five_seeds_complete': len(selected) == 5}
            for field in reference.MEAN_FIELDS:
                values = [row[field] for row in selected if row.get(field) is not None]
                mean[field] = sum(values) / len(values) if values else None
            means.append(mean)
    reference.write_rows(OUTPUT_ROOT / 'mean_by_constellation_algorithm.csv', means)


def training_finished(process, descendants):
    """Release training slots only after its entire process tree has exited."""
    if process.is_running():
        try:
            descendants.update((child.pid, child) for child in process.children(recursive=True))
        except psutil.NoSuchProcess:
            pass  # The manager may exit between the liveness check and enumeration.
    if hop_training.STATUS_PATH.exists():
        status = read_json(hop_training.STATUS_PATH)
        failures = {key: item for key, item in status.get('results', {}).items()
                    if item.get('exit_code') != 0}
        if failures:
            raise RuntimeError(f'2/4-hop training failed: {failures}')
    if process.is_running() or any(child.is_running() for child in descendants.values()):
        return False
    status = read_json(hop_training.STATUS_PATH)
    if (status.get('status') != 'complete' or status.get('expected_models') != 10
            or status.get('completed_models') != 10
            or status.get('information_hops') != [2, 4]
            or status.get('training_seeds') != list(SEEDS)):
        raise RuntimeError('2/4-hop training exited without a complete ten-model status')
    for algorithm in HOPS:
        for seed in SEEDS:
            select_model(algorithm, seed)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_acceleration_argument(parser)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--algorithm', choices=ALGORITHMS)
    parser.add_argument('--constellation', choices=CONSTELLATIONS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status-only', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--training-pid', type=int,
                        help='Share 20 slots with the live ten-worker 2/4-hop training manager')
    args = parser.parse_args()
    configure_acceleration(args.acceleration == 'on')
    if not 1 <= args.workers <= 20:
        parser.error('--workers must be between 1 and 20')
    if args.training_pid and (args.workers != 20 or args.status_only or args.rerun):
        parser.error('--training-pid requires --workers 20 without --status-only or --rerun')
    os.chdir(PROJECT_ROOT)
    algorithms = (args.algorithm,) if args.algorithm else ALGORITHMS
    seeds = (args.seed,) if args.seed is not None else SEEDS
    constellations = (args.constellation,) if args.constellation else CONSTELLATIONS
    cases = [(constellation, algorithm, seed) for constellation in constellations
             for algorithm in algorithms for seed in seeds]
    if args.dry_run:
        write_json(OUTPUT_ROOT / 'TEST_PLAN.json', {
            'expected_models': len(algorithms) * len(seeds), 'expected_cases': len(cases),
            'workers': args.workers, 'requires_validated_checkpoint': True,
            'copied_cases': sum(reuse_directory(*case) is not None for case in cases),
            'simulation_cases': sum(reuse_directory(*case) is None for case in cases),
            'cases': [build_case_config(*case) for case in cases],
        })
        print(f'DRY_RUN: {len(cases)} cases; {OUTPUT_ROOT / "TEST_PLAN.json"}')
        return
    if baseline.lock_owner_is_live(LOCK_PATH):
        raise SystemExit(f'Another test manager is running: {LOCK_PATH}')
    training, descendants = None, {}
    if args.training_pid:
        training = psutil.Process(args.training_pid)
        command = training.cmdline()
        if ('train_isacr_2_4_5seed.py' not in [Path(part).name for part in command]
                or '--workers' not in command or command[command.index('--workers') + 1] != '10'):
            raise SystemExit('Expected the live ten-worker 2/4-hop training manager')
        descendants.update((child.pid, child) for child in training.children(recursive=True))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.unlink(missing_ok=True)
    with LOCK_PATH.open('x', encoding='ascii') as stream:
        stream.write(str(os.getpid()))
    try:
        ready, errors, rows, jobs = {}, {}, [], []
        pending = set(cases)
        running, completed = set(), set()
        released = training is None
        pool_workers = args.workers if released else 10

        def prepare_cases():
            for constellation, algorithm, seed in cases:
                case = (constellation, algorithm, seed)
                if case not in pending or (not released and algorithm in HOPS):
                    continue
                key = (algorithm, seed)
                if key not in ready:
                    ready[key] = select_model(algorithm, seed)
                checkpoint = ready[key]
                if not args.rerun and case_complete(*case, checkpoint):
                    rows.append(read_json(case_directory(*case) / 'summary.json'))
                    completed.add(case)
                elif reuse_directory(*case) is not None:
                    if not args.status_only:
                        rows.append(copy_case(*case, checkpoint))
                        completed.add(case)
                else:
                    jobs.append((*case, str(checkpoint), args.rerun))
                pending.remove(case)

        def save_status():
            write_json(OUTPUT_ROOT / 'STATUS.json', {
                'status': 'complete' if len(rows) == len(cases) else 'partial',
                'expected_cases': len(cases), 'completed_cases': len(rows),
                'expected_models': len(algorithms) * len(seeds), 'ready_models': len(ready),
                'workers': pool_workers, 'errors': errors,
                'training_pid': args.training_pid, 'training_slots_released': released,
                'pending_cases': len(pending) + len(jobs), 'running_cases': len(running),
            })

        executor, futures = None, {}
        try:
            prepare_cases()
            save_status()
            while not args.status_only and (pending or jobs or futures):
                if not released and training_finished(training, descendants):
                    released = True
                    prepare_cases()
                    print('Training complete and models validated; draining ten-worker test pool before expansion', flush=True)
                # Drain and close the old pool before creating the larger pool.
                expanding = released and pool_workers != args.workers
                if expanding and not futures:
                    if executor is not None:
                        executor.shutdown(wait=True)
                        executor = None
                    pool_workers = args.workers
                    expanding = False
                while jobs and not expanding and len(futures) < pool_workers:
                    if executor is None:
                        executor = ProcessPoolExecutor(max_workers=pool_workers, initializer=baseline.initialize_worker)
                    job = jobs.pop(0)
                    case = job[:3]
                    if case in running or case in completed:
                        raise RuntimeError(f'Duplicate case submission: {case}')
                    futures[executor.submit(run_case, job)] = job
                    running.add(case)
                save_status()
                if not futures:
                    if pending:
                        time.sleep(5)
                    continue
                done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    running.remove(job[:3])
                    rows.append(future.result())
                    completed.add(job[:3])
                    print(f'Completed: {job[:3]}', flush=True)
                save_status()
        except Exception as error:
            errors['manager'] = str(error)
            try:
                save_status()
            except OSError as status_error:
                print(f'Could not publish failure status: {status_error}; original error: {error}', flush=True)
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        if not args.status_only:
            write_aggregates(rows)
        save_status()
        print(f'READY_MODELS={len(ready)}/{len(algorithms) * len(seeds)} COMPLETE_CASES={len(rows)}/{len(cases)}')
        for key, error in errors.items():
            print(f'NOT_READY {key}: {error}')
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
