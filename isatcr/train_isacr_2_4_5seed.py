"""Train 2-hop and 4-hop ISACR, each with seeds 0-4 (ten models).

Shared training settings come from train_4_5seed.py. --dry-run writes only
configuration snapshots; it never creates a training pool or touches models.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import train_4_5seed as baseline


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / 'training_process_data/v1/model'
STATUS_ROOT = MODEL_ROOT / 'isacr_2_4_5seed_training_status'
LOCK_PATH = STATUS_ROOT / 'TRAINING_RUNNING.lock'
STATUS_PATH = STATUS_ROOT / 'TRAINING_STATUS.json'
SEEDS = baseline.SEEDS
HOPS = (2, 4)
WORKERS = 10
MAX_WORKERS = 10
write_json = baseline.write_json
last_step = baseline.last_step
SOURCE_HASHES = {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                 for name in ('train_isacr_2_4_5seed.py',
                              'train_4_5seed.py', 'train_ISACR_self_elimination.py',
                              'SatelliteNetworkSimulator_Computing.py', 'Base_Agents.py',
                              'training_runner.py', 'RL_environment_for_computing.py')}


@contextmanager
def isacr_hop_environment(hops):
    """Install variable-depth ISACR aggregation only inside this training job."""
    if hops not in (2, 3, 4):
        raise ValueError('ISACR information_hops must be 2, 3, or 4')
    import numpy as np
    import SatelliteNetworkSimulator_Computing as computing

    original = computing.Satellite_with_Computing
    class ISACRHopSatellite(computing.Satellite_with_Computing):
        information_hops = 3

        @property
        def resource_scales(self):
            return [1] + [4 * 3 ** (layer - 1)
                          for layer in range(1, self.information_hops)]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            initial = [value * scale for scale in self.resource_scales
                       for value in (0, 1, 0, 0, 0, 0)]
            self.current_state = initial[:]
            self.neighbor_states = {neighbor: initial[:] for neighbor in self.neighbors}

        def _empty_neighbor_state_chunk(self):
            return [1, 0, 0, 1, 1, self._local_fault_compute_score()] * self.information_hops + [1, 2]

        def _fault_neighbor_state(self):
            return [value for scale in self.resource_scales
                    for value in [1, 0, 0, 1, 1, scale * self._local_fault_compute_score()]]

        def add_neighbor(self, neighbor):
            is_new = neighbor not in self.neighbors
            super().add_neighbor(neighbor)
            if is_new:
                self.neighbor_states[neighbor] = [
                    value * scale for scale in self.resource_scales
                    for value in (0, 1, 0, 0, 0, 0)
                ]

        def neighbor_has_resource_state(self, neighbor):
            return (super().neighbor_has_resource_state(neighbor)
                    and len(self.neighbor_states.get(neighbor, ())) == 6 * self.information_hops)

        def normalized_resources(self, state):
            if len(state) != 6 * self.information_hops:
                raise ValueError('ISACR resource message has an inconsistent hop depth')
            return [state[layer * 6 + index] / scale
                    for layer, scale in enumerate(self.resource_scales)
                    for index in range(6)]

        def fault_fill(self, normalized_states):
            fill = []
            clip = computing._clip_normalized
            multiplier = self.fault_fill_multiplier
            for layer in range(self.information_hops):
                if normalized_states:
                    mean = [sum(state[layer * 6 + index] for state in normalized_states)
                            / len(normalized_states) for index in range(6)]
                    ability = clip(mean[2] / multiplier) if self.fault_compute_double_mean else 0
                    wait = clip(mean[3] * multiplier) if self.fault_compute_double_mean else 1
                    transmission = clip(mean[4] * (multiplier if self.fault_compute_double_mean else 2))
                    effective = clip(mean[5] / multiplier) if self.fault_compute_double_mean else 0
                    fill.extend([1, 0, ability, wait, transmission, effective])
                else:
                    fill.extend([1, 0, 0, 1, 1, self._local_fault_compute_score()])
            return fill

        def aggregate_message(self, local, states, recipient=None):
            """Shift prior neighbor layers; omit recipient also from fault means."""
            if self.disable_self_feature_elimination or not self.historical_self_feature_elimination:
                recipient = None
            values = [state for name, state in states.items() if name != recipient]
            fill = self.fault_fill(values)
            # Excluding a live recipient leaves three possible branches. Pending
            # links use the same three-branch fallback as the existing 3-hop code.
            denominator = 4 if recipient is None else 3
            missing = denominator - len(values)
            message = list(local)
            for layer, scale in enumerate(self.resource_scales[1:]):
                message.extend(scale * (
                    sum(state[layer * 6 + index] for state in values)
                    + missing * fill[layer * 6 + index]
                ) / denominator for index in range(6))
            return message

        def state_exchanger(self):
            while self.active:
                yield from self._wait_for_state_exchange()
                if not self.active:
                    break
                states = {neighbor: self.normalized_resources(state)
                          for neighbor, state in self._aggregation_resource_states().items()}
                local = self.current_gnn_raw_state()[:5]
                local.append(self._normalized_compute_completion_time())
                self.current_state = self.aggregate_message(local, states)
                for recipient in sorted(set(self.neighbors) | self.pending_state_neighbors):
                    message = self.aggregate_message(local, states, recipient)
                    self.env.process(self.propagator.send_state(self.name, recipient, message))

        def get_current_state(self, destination, hops, is_computed, mission_state):
            self._refresh_transmission_features()
            resources = {neighbor: self.normalized_resources(self.neighbor_states[neighbor])
                         for neighbor in self.neighbors if self.neighbor_has_resource_state(neighbor)}
            fill = self.fault_fill(list(resources.values()))
            state = []
            for neighbor in self.action_neighbors:
                if neighbor is None:
                    slot = fill[:] + [1, 2]
                elif neighbor in resources:
                    slot = list(resources[neighbor]) + [
                        self._normalized_link_transmission_wait(neighbor),
                        self.neighbor_hops[neighbor].get(destination, 2 * self.max_hop) / self.max_hop,
                    ]
                else:
                    slot = fill[:] + [1, 2]
                if is_computed:
                    for layer in range(self.information_hops):
                        slot[layer * 6 + 2:layer * 6 + 4] = [0., 0.]
                        slot[layer * 6 + 5] = 0.
                state.extend(slot)
            local = self.current_gnn_raw_state()[:4]
            local.append(self._normalized_compute_completion_time())
            if is_computed:
                local[2:5] = [0., 0., 0.]
            return np.asarray(state + local + list(mission_state)
                              + [hops / self.max_hop, is_computed])

    class HopSatellite(ISACRHopSatellite):
        information_hops = hops
    computing.Satellite_with_Computing = HopSatellite
    try:
        yield
    finally:
        computing.Satellite_with_Computing = original


def variant_name(hops):
    return f'ISACR_{hops}hop'


def unit_directory(hops, seed):
    return STATUS_ROOT / variant_name(hops) / f'seed{seed}'


def build_training_config(hops, seed):
    if hops not in HOPS:
        raise ValueError(f'Unknown hop count: {hops}')
    config = baseline.build_training_config('ISACR_self_elimination', seed)
    name = variant_name(hops)
    config['general']['algorithm_name'] = name
    config['agent']['neighbors_dim'] = 4 * 6 * hops
    config['agent']['model_path'] = f'./training_process_data/v1/model/{name}/{name}_seed{seed}.pth'
    config['environment']['isacr_information_hops'] = hops
    config['environment']['save_training_data'] = (
        f'v1/model/{name}/train_{name}_seed{seed}_lr'
        f'{str(config["agent"]["learning_rate"]).replace(".", "p")}'
        f'_gamma{str(config["agent"]["gamma"]).replace(".", "p")}'
        f'_repeat{config["agent"]["repeat"]}.txt')
    return config


def contract_for(config):
    return {'config': json.loads(json.dumps(config)), 'source_sha256': SOURCE_HASHES}


def reward_checkpoint(hops, seed):
    name = variant_name(hops)
    matches = [path for path in (MODEL_ROOT / name).glob(f'{name}_seed{seed}_train_reward_best_*')
               if path.suffix == '.pth']
    return matches[0] if len(matches) == 1 else None


def checkpoint_matches(checkpoint, config):
    if checkpoint is None:
        return False
    from training_runner import load_checkpoint_training_config
    try:
        metadata = load_checkpoint_training_config(checkpoint)
    except (OSError, ValueError):
        return False
    return all(json.loads(json.dumps(metadata[section]))
               == json.loads(json.dumps(config[section]))
               for section in ('general', 'agent', 'environment'))


def completed_result(hops, seed):
    directory = unit_directory(hops, seed)
    if not (directory / 'process_result.json').is_file():
        return None
    result = json.loads((directory / 'process_result.json').read_text(encoding='utf-8'))
    contract_path = directory / 'contract.json'
    if not contract_path.is_file():
        return None
    stored_contract = json.loads(contract_path.read_text(encoding='utf-8'))
    expected = build_training_config(hops, seed)
    if stored_contract != contract_for(expected) or result.get('exit_code') != 0:
        return None
    checkpoint = reward_checkpoint(hops, seed)
    log = PROJECT_ROOT / 'training_process_data' / expected['environment']['save_training_data']
    from training_runner import find_latest_checkpoint
    latest = find_latest_checkpoint(PROJECT_ROOT / expected['agent']['model_path'])
    if (last_step(log) == expected['general']['duration']
            and checkpoint_matches(checkpoint, expected)
            and checkpoint_matches(latest, expected)):
        return result
    return None


def run_job(job):
    hops, seed = job
    config = build_training_config(hops, seed)
    directory = unit_directory(hops, seed)
    result = {'algorithm': variant_name(hops), 'information_hops': hops,
              'seed': seed, 'exit_code': 1, 'started_at_epoch': time.time()}
    try:
        from training_runner import run_training, find_latest_checkpoint
        write_json(directory / 'config.json', config)
        write_json(directory / 'contract.json', contract_for(config))
        with isacr_hop_environment(hops):
            run_training(config)
        log = PROJECT_ROOT / 'training_process_data' / config['environment']['save_training_data']
        reward = reward_checkpoint(hops, seed)
        latest = find_latest_checkpoint(PROJECT_ROOT / config['agent']['model_path'])
        result.update(final_step=last_step(log), training_log=str(log),
                      reward_checkpoint=str(reward) if reward else None,
                      latest_checkpoint=str(latest) if latest else None)
        if (last_step(log) != config['general']['duration']
                or not checkpoint_matches(reward, config)
                or not checkpoint_matches(latest, config)):
            result.update(exit_code=2, error='training output failed final-step/checkpoint validation')
        else:
            result['exit_code'] = 0
    except Exception:
        result['error'] = traceback.format_exc()
    result['finished_at_epoch'] = time.time()
    write_json(directory / 'process_result.json', result)
    return result


def save_status(results, jobs, workers, active, status='running'):
    completed = [key for key, item in results.items() if item['exit_code'] == 0]
    write_json(STATUS_PATH, {'status': status, 'workers': workers,
               'expected_models': len(jobs), 'completed_models': len(completed),
               'training_seeds': sorted({seed for _, seed in jobs}),
               'information_hops': sorted({hops for hops, _ in jobs}),
               'completed': completed, 'active': active, 'results': results,
               'updated_at_epoch': time.time()})


def job_key(job):
    hops, seed = job
    return f'{variant_name(hops)}/seed{seed}'


def main():
    from training_runner import prepare_reproducible_process

    prepare_reproducible_process()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hops', type=int, nargs='+', choices=HOPS, default=HOPS)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f'--workers must be between 1 and {MAX_WORKERS}')
    if len(set(args.hops)) != len(args.hops) or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('Hop counts and seeds must be unique; seeds must be non-negative')
    os.chdir(PROJECT_ROOT)
    jobs = tuple((hops, seed) for hops in args.hops for seed in args.seeds)
    if args.dry_run:
        plan = [contract_for(build_training_config(*job)) for job in jobs]
        write_json(STATUS_ROOT / 'TRAINING_PLAN.json', {'expected_models': len(jobs), 'jobs': plan})
        print(f'DRY_RUN: {len(jobs)} models; {STATUS_ROOT / "TRAINING_PLAN.json"}')
        return
    if baseline.lock_owner_is_live(LOCK_PATH):
        raise SystemExit('another 2/4-hop training manager is running or suspended')
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    results = {job_key(job): result for job in jobs if not args.rerun
               if (result := completed_result(*job)) is not None}
    pending = [job for job in jobs if job_key(job) not in results]
    conflicts = [job_key(job) for job in pending
                 if (unit_directory(*job) / 'process_result.json').exists()
                 or any((MODEL_ROOT / variant_name(job[0])).glob(f'{variant_name(job[0])}_seed{job[1]}*'))]
    if conflicts and not args.rerun:
        raise SystemExit(f'Existing outputs failed validation: {conflicts}; use --rerun to archive and retrain')
    if LOCK_PATH.exists():
        # Only this launcher's stale lock; live owners were rejected above.
        LOCK_PATH.unlink()
    with LOCK_PATH.open('x', encoding='ascii') as file:
        file.write(str(os.getpid()))
    try:
        save_status(results, jobs, args.workers, [job_key(job) for job in pending])
        with ProcessPoolExecutor(max_workers=args.workers, initializer=baseline.initialize_worker) as executor:
            futures = {executor.submit(run_job, job): job for job in pending}
            remaining = set(pending)
            for future in as_completed(futures):
                job = futures[future]
                results[job_key(job)] = future.result()
                remaining.remove(job)
                save_status(results, jobs, args.workers, [job_key(item) for item in sorted(remaining)])
                print(json.dumps(results[job_key(job)], ensure_ascii=False), flush=True)
        final = 'complete' if len(results) == len(jobs) and all(result['exit_code'] == 0 for result in results.values()) else 'partial'
        save_status(results, jobs, args.workers, [], final)
    finally:
        LOCK_PATH.unlink()


if __name__ == '__main__':
    main()
