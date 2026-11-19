"""Train five coefficients for seeds 0-4."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

from train_4_5seed import (
    build_training_config as build_isacr_training_config,
    TRAINING_SCENARIO_SEED,
    TRAINING_DURATION,
    TRAINING_REWARD_SAVE_START_STEP,
    TRAINING_EDGE_P,
    TRAINING_EDGE_Q,
    TRAINING_LOAD,
    TRAINING_FAULT_FILL_MULTIPLIER,
)
from train_ISACR_self_elimination import CONFIG as ISACR_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = (
    PROJECT_ROOT / 'training_process_data' / 'v1' / 'model'
    / 'ISACR_fault_self_elimination'
)
UPDATE_CYCLE = ISACR_CONFIG['agent']['update_cycle']
STATUS_ROOT = MODEL_ROOT / 'five_coefficient_training_status'
LOCK_PATH = STATUS_ROOT / 'TRAINING_RUNNING.lock'
STATUS_PATH = STATUS_ROOT / 'TRAINING_STATUS.json'

TRAINING_SEEDS = (0, 1, 2, 3, 4)
FILL_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5, 3.0)
# Copy the four-algorithm coefficient; train the other coefficients.
COPIED_MULTIPLIER = TRAINING_FAULT_FILL_MULTIPLIER
TRAINING_MULTIPLIERS = tuple(value for value in FILL_MULTIPLIERS if value != COPIED_MULTIPLIER)
TRAINING_WORKERS = 10
JOBS = tuple(
    (multiplier, seed)
    for multiplier in TRAINING_MULTIPLIERS
    for seed in TRAINING_SEEDS
)


def copy_stage1_coefficient():
    """Copy the matching coefficient from the completed four-algorithm ISACR run."""
    if COPIED_MULTIPLIER not in FILL_MULTIPLIERS:
        return

    from training_runner import load_checkpoint_training_config, checkpoint_content_hashes

    source_model_root = PROJECT_ROOT / 'training_process_data' / 'v1' / 'model'
    source_root = source_model_root / 'four_5seed_training_status' / 'ISACR_self_elimination'
    for seed in TRAINING_SEEDS:
        if completed_result(COPIED_MULTIPLIER, seed, copied=True) is not None:
            continue
        if output_paths(COPIED_MULTIPLIER, seed)['directory'].exists() and any(
            output_paths(COPIED_MULTIPLIER, seed)['directory'].iterdir()
        ):
            raise RuntimeError(
                f'coefficient {COPIED_MULTIPLIER} seed {seed}: existing copy failed validation; '
                'preserve and inspect it before replacing'
            )
        source_status = source_root / f'seed{seed}' / 'process_result.json'
        source_config = source_root / f'seed{seed}' / 'config.json'
        if not source_status.is_file() or not source_config.is_file():
            raise RuntimeError(
                f'coefficient {COPIED_MULTIPLIER} source is incomplete for seed {seed}: '
                f'{source_status}'
            )
        result = json.loads(source_status.read_text(encoding='utf-8'))
        config = json.loads(source_config.read_text(encoding='utf-8'))
        source_checkpoint = Path(result.get('reward_checkpoint') or '')
        source_log = Path(result.get('training_log') or '')
        if (
            result.get('exit_code') != 0
            or result.get('final_step') != TRAINING_DURATION
            or config.get('environment', {}).get('target_load_tasks_per_second') != TRAINING_LOAD['target_load_tasks_per_second']
            or config.get('environment', {}).get('edge_p') != TRAINING_EDGE_P
            or config.get('environment', {}).get('edge_q') != TRAINING_EDGE_Q
            or not source_checkpoint.is_file()
            or not source_log.is_file()
            or last_step(source_log) != TRAINING_DURATION
        ):
            raise RuntimeError(f'coefficient {COPIED_MULTIPLIER} source failed validation for seed {seed}')
        checkpoint_config = load_checkpoint_training_config(source_checkpoint)
        if any(checkpoint_config[section] != config[section]
               for section in ('general', 'agent', 'environment')):
            raise RuntimeError(f'coefficient {COPIED_MULTIPLIER} source metadata mismatch for seed {seed}')
        paths = output_paths(COPIED_MULTIPLIER, seed)
        expected = build_training_config(COPIED_MULTIPLIER, seed)
        aligned_source = json.loads(json.dumps(checkpoint_config))
        aligned_source['general']['algorithm_name'] = expected['general']['algorithm_name']
        aligned_source['agent']['model_path'] = expected['agent']['model_path']
        aligned_source['environment']['save_training_data'] = expected['environment']['save_training_data']
        validate_training_config(aligned_source, COPIED_MULTIPLIER, seed)
        paths['directory'].mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, paths['model_path'])
        shutil.copy2(source_log, paths['training_log'])
        config = aligned_source
        write_json(paths['config'], config)
        write_json(Path(str(paths['model_path']) + '.training.json'), {
            'config': config,
            'checkpoint_sha256': checkpoint_content_hashes(paths['model_path']),
        })
        copied_result = dict(result)
        copied_result.update({
            'coefficient': COPIED_MULTIPLIER,
            'copied': True,
            'source_algorithm': 'ISACR_self_elimination',
            'source_checkpoint': str(source_checkpoint),
            'source_model_root': str(source_model_root),
            'training_log': str(paths['training_log']),
            'reward_checkpoint': str(paths['model_path']),
            'latest_checkpoint': str(paths['model_path']),
        })
        write_json(paths['process_result'], copied_result)


def configure_worker():
    """Configure PyTorch once when a ProcessPool worker is created."""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def number_tag(value):
    return str(float(value)).replace('.', 'p')


def output_paths(multiplier, seed):
    tag = number_tag(multiplier)
    output_dir = MODEL_ROOT / f'seed{seed}' / f'coefficient_{tag}'
    stem = (
        f'ISACR_fault_self_elimination_seed{seed}_fill{tag}'
        f'_lr{str(ISACR_CONFIG["agent"]["learning_rate"]).replace(".", "p")}'
        '_gamma0p99_repeat10'
    )
    return {
        'directory': output_dir,
        'model_path': output_dir / f'{stem}.pth',
        'training_log': output_dir / f'train_{stem}.txt',
        'console_log': output_dir / f'train_{stem}_console.txt',
        'config': output_dir / 'training_config.json',
        'process_result': output_dir / 'process_result.json',
    }


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


def reward_checkpoint(multiplier, seed):
    paths = output_paths(multiplier, seed)
    matches = [
        path for path in paths['directory'].glob(
            f'ISACR_fault_self_elimination_seed{seed}_train_reward_best_*'
        )
        if not path.name.endswith('.training.json')
    ]
    return matches[0] if len(matches) == 1 else None


def build_training_config(multiplier, seed):
    config = build_isacr_training_config('ISACR_self_elimination', seed)
    paths = output_paths(multiplier, seed)
    config['general']['algorithm_name'] = 'ISACR_fault_self_elimination'
    config['agent'].update(
        model_path=str(paths['model_path']),
        update_cycle=UPDATE_CYCLE,
    )
    config['environment'].update(
        fault_fill_multiplier=multiplier,
        save_training_data=str(paths['training_log'].relative_to(
            PROJECT_ROOT / 'training_process_data'
        )),
    )
    return config


def validate_training_config(config, multiplier, seed):
    expected = build_training_config(multiplier, seed)
    for section in ('general', 'agent', 'environment'):
        if json.loads(json.dumps(config[section])) != json.loads(json.dumps(expected[section])):
            raise ValueError(f'Five-coefficient/current ISACR setting mismatch: {section}')


def completed_result(multiplier, seed, copied=False):
    """Return a validated prior result so completed models are not retrained."""
    paths = output_paths(multiplier, seed)
    result_path = paths['process_result']
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding='utf-8'))
        config = json.loads(paths['config'].read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return None
    log_path = result.get('training_log')
    checkpoint = result.get('reward_checkpoint')
    from train_ISACR_self_elimination import CONFIG
    if (
        result.get('exit_code') != 0
        or config.get('agent', {}).get('neighbors_dim') != CONFIG['agent']['neighbors_dim']
        or bool(result.get('copied')) != copied
        or result.get('final_step') != TRAINING_DURATION
        or config.get('general', {}).get('random_seed') != seed
        or config.get('general', {}).get('scenario_seed') != TRAINING_SCENARIO_SEED
        or config.get('general', {}).get('duration') != TRAINING_DURATION
        or config.get('general', {}).get('warmup_steps') != CONFIG['general']['warmup_steps']
        or config.get('general', {}).get('training_reward_save_start_step') != TRAINING_REWARD_SAVE_START_STEP
        or config.get('agent', {}).get('update_cycle') != UPDATE_CYCLE
        or config.get('environment', {}).get('fault_fill_multiplier') != multiplier
        or config.get('environment', {}).get('edge_p') != TRAINING_EDGE_P
        or config.get('environment', {}).get('edge_q') != TRAINING_EDGE_Q
        or any(config.get('environment', {}).get(key) != value for key, value in TRAINING_LOAD.items())
        or not log_path
        or last_step(Path(log_path)) != TRAINING_DURATION
        or not checkpoint
        or not Path(checkpoint).is_file()
    ):
        return None
    try:
        from training_runner import load_checkpoint_training_config
        validate_training_config(config, multiplier, seed)
        checkpoint_config = load_checkpoint_training_config(checkpoint)
        if any(checkpoint_config[section] != config[section]
               for section in ('general', 'agent', 'environment')):
            return None
    except (OSError, ValueError):
        return None
    return result


def run_job(job):
    multiplier, seed = job
    paths = output_paths(multiplier, seed)
    result = {
        'coefficient': multiplier,
        'seed': seed,
        'exit_code': 1,
        'started_at_epoch': time.time(),
    }
    try:
        from training_runner import run_training

        config = build_training_config(multiplier, seed)
        validate_training_config(config, multiplier, seed)
        write_json(paths['config'], config)

        paths['directory'].mkdir(parents=True, exist_ok=True)
        with paths['console_log'].open('w', encoding='utf-8', buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                run_training(config)

        checkpoint = reward_checkpoint(multiplier, seed)
        result.update({
            'exit_code': 0,
            'final_step': last_step(paths['training_log']),
            'model_path': str(paths['model_path']),
            'training_log': str(paths['training_log']),
            'reward_checkpoint': str(checkpoint) if checkpoint else None,
        })
        if result['final_step'] != TRAINING_DURATION or checkpoint is None:
            result['exit_code'] = 2
            result['error'] = 'training output failed final-step/checkpoint validation'
    except Exception:
        result['error'] = traceback.format_exc()
    result['finished_at_epoch'] = time.time()
    write_json(paths['process_result'], result)
    return result


def save_status(results, workers=TRAINING_WORKERS):
    completed = [
        key for key, value in results.items() if value.get('exit_code') == 0
    ]
    trained = [
        key for key, value in results.items()
        if value.get('exit_code') == 0 and not value.get('copied')
    ]
    expected_models = len(FILL_MULTIPLIERS) * len(TRAINING_SEEDS)
    write_json(STATUS_PATH, {
        'status': 'complete' if len(completed) == expected_models else 'running',
        'coefficients': FILL_MULTIPLIERS,
        'trained_coefficients': TRAINING_MULTIPLIERS,
        'training_seeds': TRAINING_SEEDS,
        'workers': workers,
        'expected_models': expected_models,
        'expected_trained_models': len(JOBS),
        'trained_models': len(trained),
        'copied_models': sum(1 for value in results.values()
                             if value.get('exit_code') == 0 and value.get('copied')),
        'completed_models': len(completed),
        'completed': completed,
        'results': results,
        'updated_at_epoch': time.time(),
    })


def clear_stale_process_records():
    """Remove per-model records left by an interrupted previous run."""
    for multiplier, seed in JOBS:
        output_paths(multiplier, seed)['process_result'].unlink(missing_ok=True)


def main():
    from training_runner import prepare_reproducible_process

    prepare_reproducible_process()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=TRAINING_WORKERS)
    parser.add_argument(
        '--rerun', action='store_true',
        help='Retrain the dedicated coefficient models for seeds 0-4.',
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')

    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'training is already running: {LOCK_PATH}')
    with os.fdopen(descriptor, 'w', encoding='ascii') as stream:
        stream.write(str(os.getpid()))

    copy_stage1_coefficient()
    results = {
        f'coefficient_{number_tag(multiplier)}/seed{seed}': result
        for multiplier in FILL_MULTIPLIERS
        for seed in TRAINING_SEEDS
        if multiplier == COPIED_MULTIPLIER or not args.rerun
        if (result := completed_result(multiplier, seed, copied=multiplier == COPIED_MULTIPLIER)) is not None
    }
    pending_jobs = [
        job for job in JOBS
        if f'coefficient_{number_tag(job[0])}/seed{job[1]}' not in results
    ]
    if not args.rerun:
        conflicts = [
            f'coefficient_{number_tag(multiplier)}/seed{seed}'
            for multiplier, seed in pending_jobs
            if output_paths(multiplier, seed)['directory'].exists()
            and any(output_paths(multiplier, seed)['directory'].iterdir())
        ]
        if conflicts:
            LOCK_PATH.unlink(missing_ok=True)
            raise SystemExit(
                f'Existing training outputs failed validation: {conflicts}. '
                'Explicit --rerun archives old models and logs.'
            )
    save_status(results, args.workers)
    try:
        if pending_jobs:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=configure_worker,
            ) as executor:
                futures = {
                    executor.submit(run_job, job): job for job in pending_jobs
                }
                for future in as_completed(futures):
                    multiplier, seed = futures[future]
                    result = future.result()
                    key = f'coefficient_{number_tag(multiplier)}/seed{seed}'
                    results[key] = result
                    save_status(results, args.workers)
                    print(json.dumps(result, ensure_ascii=False), flush=True)

        status = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
        status['status'] = (
            'complete'
            if len(results) == len(FILL_MULTIPLIERS) * len(TRAINING_SEEDS)
            and all(item.get('exit_code') == 0 for item in results.values())
            else 'partial'
        )
        write_json(STATUS_PATH, status)
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
