"""Train five ISACR ablations across five seeds with up to fifteen workers."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
import psutil

import train_4_5seed as baseline
import training_runner
from ablation.training_entry import TRAINING_SEEDS, VARIANT_NAMES, build_config, _json_ready
from training_runner import find_latest_checkpoint, load_checkpoint_training_config, run_training


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'model'
STATUS_ROOT = MODEL_ROOT / 'ablation_training_status'
LOCK_PATH = STATUS_ROOT / 'TRAINING_RUNNING.lock'
STATUS_PATH = STATUS_ROOT / 'TRAINING_STATUS.json'
MAX_WORKERS = 15
write_json = baseline.write_json
last_step = baseline.last_step
IMPLEMENTATION_SHA256 = hashlib.sha256(b''.join(
    (PROJECT_ROOT / name).read_bytes() for name in (
        'train_ablation.py', 'train_4_5seed.py', 'train_ISACR_self_elimination.py',
        'ablation/aggregation_models.py', 'ablation/message_aggregation.py', 'ablation/training_entry.py',
        'ablation/train_graphsage.py', 'SatelliteNetworkSimulator_Computing.py',
        'SatelliteNetworkSimulator_Beta.py', 'Base_Agents.py', 'training_runner.py', 'replay_storage.py',
    )
)).hexdigest()


def paths_for(encoder, seed):
    config = build_config(encoder, seed)
    directory = MODEL_ROOT / VARIANT_NAMES[encoder] / f'seed{seed}'
    log = PROJECT_ROOT / 'training_process_data' / config['environment']['save_training_data']
    return config, directory, log


def checkpoint_exists(path, config):
    if config['agent']['mode'] == 'GNN_DQN':
        return path.is_dir() and all((path / name).is_file() for name in ('gnn.pth', 'qnet.pth'))
    return path.is_file()


def reward_checkpoint(encoder, seed):
    config, directory, _ = paths_for(encoder, seed)
    matches = [path for path in directory.glob(f'{VARIANT_NAMES[encoder]}_seed{seed}_train_reward_best_*')
               if not path.name.endswith('.training.json') and checkpoint_exists(path, config)]
    return matches[0] if len(matches) == 1 else None


def checkpoint_matches(path, config):
    if path is None or not checkpoint_exists(path, config):
        return False
    try:
        saved = load_checkpoint_training_config(path)
        expected = _json_ready(config)
        return all(saved[section] == expected[section] for section in ('general', 'agent', 'environment'))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def completed_result(encoder, seed):
    config, directory, log = paths_for(encoder, seed)
    try:
        result = json.loads((directory / 'process_result.json').read_text(encoding='utf-8'))
        saved = json.loads((directory / 'training_config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or not isinstance(saved, dict):
        return None
    if (result.get('encoder') != encoder
            or result.get('algorithm') != VARIANT_NAMES[encoder]
            or result.get('seed') != seed
            or result.get('exit_code') != 0
            or result.get('implementation_sha256') != IMPLEMENTATION_SHA256
            or result.get('final_step') != config['general']['duration']
            or saved != _json_ready(config)
            or last_step(log) != config['general']['duration']):
        return None
    reward = reward_checkpoint(encoder, seed)
    latest = find_latest_checkpoint(PROJECT_ROOT / config['agent']['model_path'])
    if (result.get('reward_checkpoint') != str(reward)
            or result.get('latest_checkpoint') != str(latest)
            or not checkpoint_matches(reward, config)
            or not checkpoint_matches(latest, config)):
        return None
    return result


def job_complete(encoder, seed):
    return completed_result(encoder, seed) is not None


def run_job(job):
    encoder, seed = job
    config, directory, log = paths_for(encoder, seed)
    result = dict(encoder=encoder, algorithm=VARIANT_NAMES[encoder], seed=seed,
                  exit_code=1, started_at_epoch=time.time(), implementation_sha256=IMPLEMENTATION_SHA256)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        write_json(directory / 'worker_identity.json', dict(
            pid=os.getpid(), create_time=psutil.Process().create_time(),
            encoder=encoder, seed=seed, implementation_sha256=IMPLEMENTATION_SHA256))
        write_json(directory / 'training_config.json', _json_ready(config))
        with (directory / 'console_stdout.txt').open('w', encoding='utf-8') as stdout, \
                (directory / 'console_stderr.txt').open('w', encoding='utf-8') as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                original_warmup = training_runner.warmup_training_environment
                def report_warmup(env, steps, epsilon):
                    original_step = env.step
                    completed = 0
                    def step(*args, **kwargs):
                        nonlocal completed
                        value = original_step(*args, **kwargs)
                        completed += 1
                        env.print_and_save(f'Warmup progress: {completed}/{steps}')
                        return value
                    env.step = step
                    try:
                        return original_warmup(env, steps, epsilon)
                    finally:
                        env.step = original_step
                training_runner.warmup_training_environment = report_warmup
                try:
                    run_training(config)
                finally:
                    training_runner.warmup_training_environment = original_warmup
        reward = reward_checkpoint(encoder, seed)
        latest = find_latest_checkpoint(PROJECT_ROOT / config['agent']['model_path'])
        result.update(final_step=last_step(log), training_log=str(log),
                      reward_checkpoint=str(reward) if reward else None,
                      latest_checkpoint=str(latest) if latest else None)
        if (result['final_step'] == config['general']['duration']
                and checkpoint_matches(reward, config) and checkpoint_matches(latest, config)):
            result['exit_code'] = 0
        else:
            result.update(exit_code=2, error='training output failed final-step/checkpoint validation')
    except Exception:
        result['error'] = traceback.format_exc()
    result['finished_at_epoch'] = time.time()
    write_json(directory / 'process_result.json', result)
    return result


def job_key(job):
    encoder, seed = job
    return f'{VARIANT_NAMES[encoder]}/seed{seed}'


def save_status(results, jobs, workers, active, status='running'):
    completed = [key for key, result in results.items() if result['exit_code'] == 0]
    write_json(STATUS_PATH, dict(status=status, workers=workers, expected_models=len(jobs),
               pid=os.getpid(), create_time=psutil.Process().create_time(),
               completed_models=len(completed), variants=list(dict.fromkeys(VARIANT_NAMES[e] for e, _ in jobs)),
               training_seeds=sorted({seed for _, seed in jobs}), completed=completed,
               active=active, results=results, updated_at_epoch=time.time()))


def archive_outputs(jobs):
    directories = [paths_for(*job)[1] for job in jobs]
    directories = [path for path in directories if path.exists() and any(path.iterdir())]
    if not directories:
        return
    for path in directories:
        expected_parent = (MODEL_ROOT / path.parent.name).resolve()
        if path.resolve().parent != expected_parent or expected_parent.parent != MODEL_ROOT.resolve():
            raise ValueError(f'Unexpected ablation output path: {path}')
    root = MODEL_ROOT / 'restart_archives'
    root.mkdir(parents=True, exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix='ablation_', dir=root))
    for path in directories:
        target = archive / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    if STATUS_PATH.is_file():
        shutil.copy2(STATUS_PATH, archive / 'TRAINING_STATUS.json')
    print(f'Previous ablation outputs preserved: {archive}', flush=True)


def main():
    from training_runner import prepare_reproducible_process

    prepare_reproducible_process()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--algorithms', nargs='+', choices=tuple(VARIANT_NAMES.values()),
                        default=tuple(VARIANT_NAMES.values()))
    parser.add_argument('--seeds', type=int, nargs='+', default=TRAINING_SEEDS)
    parser.add_argument('--rerun', action='store_true', help='Archive selected outputs before fresh training.')
    parser.add_argument('--dry-run', action='store_true', help='Write the plan without starting training.')
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f'--workers must be between 1 and {MAX_WORKERS}')
    if (len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds)
            or len(set(args.algorithms)) != len(args.algorithms)):
        parser.error('Algorithms and seeds must be unique; seeds must be non-negative')
    os.chdir(PROJECT_ROOT)
    jobs = tuple((encoder, seed) for encoder, variant in VARIANT_NAMES.items()
                 if variant in args.algorithms for seed in args.seeds)
    if args.dry_run:
        write_json(STATUS_ROOT / 'TRAINING_PLAN.json', dict(workers=args.workers,
                   expected_models=len(jobs), training_seeds=list(args.seeds),
                   jobs=[_json_ready(build_config(*job)) for job in jobs]))
        print(f'DRY_RUN: {len(jobs)} models, {args.workers} workers')
        return
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.is_file():
        try:
            owner = psutil.Process(int(LOCK_PATH.read_text(encoding='ascii').strip()))
            if any(Path(arg).name == 'train_ablation.py' for arg in owner.cmdline()):
                raise SystemExit('another ablation training manager is running or suspended')
        except psutil.NoSuchProcess:
            pass
    LOCK_PATH.unlink(missing_ok=True)
    with LOCK_PATH.open('x', encoding='ascii') as lock:
        lock.write(str(os.getpid()))
    try:
        results = {job_key(job): result for job in jobs if not args.rerun
                   if (result := completed_result(*job)) is not None}
        pending = [job for job in jobs if job_key(job) not in results]
        conflicts = [job_key(job) for job in pending
                     if paths_for(*job)[1].exists() and any(paths_for(*job)[1].iterdir())]
        if conflicts and not args.rerun:
            raise SystemExit(f'Existing outputs failed validation: {conflicts}; use --rerun to archive and retrain')
        if args.rerun:
            archive_outputs(pending)
        save_status(results, jobs, args.workers, [job_key(job) for job in pending])
        with ProcessPoolExecutor(max_workers=args.workers, initializer=baseline.initialize_worker) as executor:
            remaining = set(pending)
            futures = {executor.submit(run_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception:
                    encoder, seed = job
                    result = dict(encoder=encoder, algorithm=VARIANT_NAMES[encoder], seed=seed,
                                  exit_code=1, error=traceback.format_exc(),
                                  implementation_sha256=IMPLEMENTATION_SHA256,
                                  finished_at_epoch=time.time())
                results[job_key(job)] = result
                remaining.remove(job)
                save_status(results, jobs, args.workers, [job_key(item) for item in sorted(remaining)])
                print(json.dumps(result, ensure_ascii=False), flush=True)
        final = 'complete' if len(results) == len(jobs) and all(r['exit_code'] == 0 for r in results.values()) else 'partial'
        save_status(results, jobs, args.workers, [], final)
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
