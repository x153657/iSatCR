"""Train selected algorithms for selected seeds with worker processes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import os
from pathlib import Path
import re
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'model'
STATUS_ROOT = MODEL_ROOT / 'four_5seed_training_status'
LOCK_PATH = STATUS_ROOT / 'TRAINING_RUNNING.lock'
OLD_LOCK_PATH = MODEL_ROOT / 'seed0_4_training_status' / 'TRAINING_RUNNING.lock'
STATUS_PATH = STATUS_ROOT / 'TRAINING_STATUS.json'
# Train the five requested seeds for every selected algorithm.
SEEDS = (0, 1, 2, 3, 4)
TRAINING_SCENARIO_SEED = 2020
WORKERS = 10
MAX_WORKERS = 10
TRAINING_DURATION = 1000
TRAINING_REWARD_SAVE_START_STEP = 900
TRAINING_EDGE_P = 0.18
TRAINING_EDGE_Q = 0.82
TRAINING_FAULT_FILL_MULTIPLIER = 2.0
TRAINING_LOAD = {
    'id': 'load_160',
    'target_load_tasks_per_second': 160,
    'poisson_rate_g': 14.57,
    'poisson_rate_s': 29.15,
}

# Algorithm logic, hyperparameters, and parameter-labelled log paths come from
# the selected training modules. This table fixes model locations for this run.
ALGORITHMS = {
    'D3QN': {
        'module': 'train_D3QN',
        'model_path': './training_process_data/v1/model/D3QN/D3QN_seed{seed}.pth',
    },
    'DDQN': {
        'module': 'train_DDQN',
        'model_path': './training_process_data/v1/model/DDQN/DDQN_seed{seed}.pth',
    },
    'PPO': {
        'module': 'train_PPO',
        'model_path': './training_process_data/v1/model/ppo/PPO_seed{seed}',
    },
    'ISACR_self_elimination': {
        'module': 'train_ISACR_self_elimination',
        'model_path': (
            './training_process_data/v1/model/ISACR_self_elimination/'
            'ISACR_self_elimination_seed{seed}.pth'
        ),
    },
}

# Queue jobs in this order; workers take the next job as soon as they are free.
TRAINING_PRIORITY = (
    'ISACR_self_elimination',
    'PPO',
    'DDQN',
    'D3QN',
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8',
    )
    temporary.replace(path)


def last_step(path):
    if not path.is_file():
        return None
    matches = re.findall(
        r'^====== step (\d+) ======$',
        path.read_text(encoding='utf-8', errors='replace'),
        re.MULTILINE,
    )
    return int(matches[-1]) if matches else None


def lock_owner_is_live(path):
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text(encoding='ascii').strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def reward_checkpoint(algorithm, seed):
    model_path = Path(ALGORITHMS[algorithm]['model_path'].format(seed=seed))
    directory = PROJECT_ROOT / model_path.parent
    # A checkpoint always has a sidecar ``.training.json`` file.  PPO uses a
    # directory checkpoint, while the other agents use a ``.pth`` file; in
    # both cases the sidecar must not count as a second checkpoint.
    matches = [
        path for path in directory.glob(f'{algorithm}_seed{seed}_train_reward_best_*')
        if not path.name.endswith('.training.json')
    ]
    return matches[0] if len(matches) == 1 else None


def initialize_worker():
    """Configure Torch once when a pool worker starts.

    ``set_num_interop_threads`` can only be called before Torch has started
    parallel work, so it must not run once per job in a reused worker.
    """
    torch = importlib.import_module('torch')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def build_training_config(algorithm, seed):
    module = importlib.import_module(ALGORITHMS[algorithm]['module'])
    config = module.seed_config(seed)
    config['general'].update(
        scenario_seed=TRAINING_SCENARIO_SEED,
        duration=TRAINING_DURATION,
        training_reward_save_start_step=TRAINING_REWARD_SAVE_START_STEP,
    )
    config['environment'].update(
        edge_p=TRAINING_EDGE_P, edge_q=TRAINING_EDGE_Q, **TRAINING_LOAD,
        fault_fill_multiplier=TRAINING_FAULT_FILL_MULTIPLIER,
    )
    config['agent']['model_path'] = ALGORITHMS[algorithm]['model_path'].format(seed=seed)
    return config


def validate_training_config(config, algorithm, seed):
    expected = build_training_config(algorithm, seed)
    if (algorithm == 'ISACR_self_elimination'
            and config['general']['algorithm_name'] == 'ISACR_fault_self_elimination'):
        # A coefficient-2 run has the same policy, with different artifact names.
        config = json.loads(json.dumps(config))
        config['general']['algorithm_name'] = expected['general']['algorithm_name']
        config['agent']['model_path'] = expected['agent']['model_path']
        config['environment']['save_training_data'] = expected['environment']['save_training_data']
    for section in ('general', 'agent', 'environment'):
        if json.loads(json.dumps(config[section])) != json.loads(json.dumps(expected[section])):
            raise ValueError(f'Four-algorithm training setting mismatch: {algorithm}.{section}')


def completed_result(algorithm, seed):
    """Return a previously validated result, or ``None`` if it must rerun."""
    unit_dir = STATUS_ROOT / algorithm / f'seed{seed}'
    result_path = unit_dir / 'process_result.json'
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding='utf-8'))
        config = json.loads((unit_dir / 'config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return None
    checkpoint = result.get('reward_checkpoint')
    training_log = result.get('training_log')
    # Older runs were marked exit_code=2 because reward_checkpoint() counted
    # the metadata sidecar as a second match.  Repair that false negative only
    # when the finished log and the actual reward-best checkpoint are present.
    if (
        result.get('exit_code') == 2
        and result.get('error') == 'training output failed final-step/checkpoint validation'
        and result.get('final_step') == TRAINING_DURATION
        and training_log
        and last_step(Path(training_log)) == TRAINING_DURATION
        and not checkpoint
    ):
        repaired_checkpoint = reward_checkpoint(algorithm, seed)
        if repaired_checkpoint is not None:
            result.update({
                'exit_code': 0,
                'reward_checkpoint': str(repaired_checkpoint),
            })
            result.pop('error', None)
            write_json(result_path, result)
            checkpoint = result['reward_checkpoint']
    expected_neighbors_dim = importlib.import_module(ALGORITHMS[algorithm]['module']).CONFIG['agent']['neighbors_dim']
    if (
        result.get('exit_code') != 0
        or config.get('agent', {}).get('neighbors_dim') != expected_neighbors_dim
        or result.get('final_step') != TRAINING_DURATION
        or config.get('general', {}).get('random_seed') != seed
        or config.get('general', {}).get('scenario_seed') != TRAINING_SCENARIO_SEED
        or config.get('general', {}).get('duration') != TRAINING_DURATION
        or config.get('general', {}).get('warmup_steps') != 151
        or config.get('general', {}).get('training_reward_save_start_step') != TRAINING_REWARD_SAVE_START_STEP
        or config.get('environment', {}).get('fault_fill_multiplier') != TRAINING_FAULT_FILL_MULTIPLIER
        or config.get('environment', {}).get('edge_p') != TRAINING_EDGE_P
        or config.get('environment', {}).get('edge_q') != TRAINING_EDGE_Q
        or any(config.get('environment', {}).get(key) != value for key, value in TRAINING_LOAD.items())
        or not checkpoint
        or not (
            Path(checkpoint).is_file()
            or (algorithm == 'PPO' and all(
                (Path(checkpoint) / name).is_file()
                for name in ('actor.pth', 'critic.pth')
            ))
        )
        or not training_log
        or last_step(Path(training_log)) != TRAINING_DURATION
    ):
        return None
    try:
        from training_runner import load_checkpoint_training_config
        validate_training_config(config, algorithm, seed)
        if result.get('copied_from_coefficient') == 2.0 and algorithm == 'ISACR_self_elimination':
            from test_5ix_5seed_5sat import load_checkpoint_training_config
        checkpoint_config = load_checkpoint_training_config(checkpoint)
        if any(checkpoint_config[section] != config[section]
               for section in ('general', 'agent', 'environment')):
            return None
    except (OSError, ValueError):
        return None
    return result


def run_job(job):
    algorithm, seed = job
    specification = ALGORITHMS[algorithm]
    unit_dir = STATUS_ROOT / algorithm / f'seed{seed}'
    result = {
        'algorithm': algorithm,
        'seed': seed,
        'exit_code': 1,
        'started_at_epoch': time.time(),
    }
    try:
        module = importlib.import_module(specification['module'])
        config = build_training_config(algorithm, seed)
        validate_training_config(config, algorithm, seed)
        write_json(unit_dir / 'config.json', config)

        module.run_training(config)
        from training_runner import find_latest_checkpoint

        log_path = (
            PROJECT_ROOT / 'training_process_data'
            / config['environment']['save_training_data']
        )
        checkpoint = reward_checkpoint(algorithm, seed)
        latest_checkpoint = find_latest_checkpoint(
            PROJECT_ROOT / config['agent']['model_path']
        )
        result.update({
            'exit_code': 0,
            'final_step': last_step(log_path),
            'training_log': str(log_path),
            'reward_checkpoint': str(checkpoint) if checkpoint else None,
            'latest_checkpoint': str(latest_checkpoint) if latest_checkpoint else None,
        })
        if (
            result['final_step'] != TRAINING_DURATION
            or checkpoint is None
            or latest_checkpoint is None
        ):
            result['exit_code'] = 2
            result['error'] = 'training output failed final-step/checkpoint validation'
    except Exception:
        result['error'] = traceback.format_exc()
    result['finished_at_epoch'] = time.time()
    write_json(unit_dir / 'process_result.json', result)
    return result


def save_status(results, jobs, workers, active=None):
    completed = [
        f"{item['algorithm']}/seed{item['seed']}"
        for item in results.values()
        if item.get('exit_code') == 0
    ]
    write_json(STATUS_PATH, {
        'status': 'complete' if len(completed) == len(jobs) else 'running',
        'workers': workers,
        'expected_models': len(jobs),
        'completed_models': len(completed),
        'training_seeds': sorted({seed for _, seed in jobs}),
        'completed': completed,
        'active': active or [],
        'results': results,
        'updated_at_epoch': time.time(),
    })


def main():
    from training_runner import prepare_reproducible_process

    prepare_reproducible_process()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--algorithms', nargs='+', choices=tuple(ALGORITHMS),
        default=tuple(ALGORITHMS),
    )
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        help='Train exactly these non-negative random seeds instead of the default seed set.',
    )
    parser.add_argument(
        '--rerun', action='store_true',
        help='Retrain selected jobs even when validated results already exist.',
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f'--workers must be between 1 and {MAX_WORKERS}')
    selected_seeds = SEEDS if args.seeds is None else tuple(args.seeds)
    if (
        not selected_seeds
        or any(seed < 0 for seed in selected_seeds)
        or len(set(selected_seeds)) != len(selected_seeds)
    ):
        parser.error('--seeds must contain unique non-negative integers')
    requested_algorithms = set(args.algorithms)
    selected_algorithms = tuple(
        algorithm
        for algorithm in TRAINING_PRIORITY
        if algorithm in requested_algorithms
    )
    jobs = tuple(
        (algorithm, seed)
        for algorithm in selected_algorithms
        for seed in selected_seeds
    )

    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    if lock_owner_is_live(LOCK_PATH) or lock_owner_is_live(OLD_LOCK_PATH):
        raise SystemExit('another four-algorithm training manager is already running')
    LOCK_PATH.write_text(str(os.getpid()), encoding='ascii')

    results = {
        f'{algorithm}/seed{seed}': result
        for algorithm, seed in jobs
        if not args.rerun
        if (result := completed_result(algorithm, seed)) is not None
    }
    pending_jobs = [
        job for job in jobs
        if f'{job[0]}/seed{job[1]}' not in results
    ]
    if not args.rerun:
        conflicts = [
            f'{algorithm}/seed{seed}' for algorithm, seed in pending_jobs
            if (STATUS_ROOT / algorithm / f'seed{seed}' / 'process_result.json').exists()
            or any((MODEL_ROOT / Path(ALGORITHMS[algorithm]['model_path']).parent.name)
                   .glob(f'{algorithm}_seed{seed}_*'))
        ]
        if conflicts:
            LOCK_PATH.unlink(missing_ok=True)
            raise SystemExit(
                f'Existing training outputs failed validation: {conflicts}. '
                'Resolve the mismatch before testing; explicit --rerun archives old outputs.'
            )
    save_status(
        results,
        jobs,
        args.workers,
        [f'{algorithm}/seed{seed}' for algorithm, seed in pending_jobs],
    )
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
        ) as executor:
            remaining = set(pending_jobs)
            futures = {
                executor.submit(run_job, job): job for job in pending_jobs
            }
            for future in as_completed(futures):
                algorithm, seed = futures[future]
                result = future.result()
                results[f'{algorithm}/seed{seed}'] = result
                remaining.discard((algorithm, seed))
                save_status(
                    results,
                    jobs,
                    args.workers,
                    [f'{item[0]}/seed{item[1]}' for item in remaining],
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)
        final_status = 'complete' if all(
            item.get('exit_code') == 0 for item in results.values()
        ) and len(results) == len(jobs) else 'partial'
        status = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
        status['status'] = final_status
        write_json(STATUS_PATH, status)
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
