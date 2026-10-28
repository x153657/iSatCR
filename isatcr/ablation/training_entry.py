"""Shared configuration and entry point for all ablation training jobs."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.aggregation_models import make_encoder_factory
from train_4_5seed import SEEDS as TRAINING_SEEDS, build_training_config as build_isacr_training_config
from training_runner import run_training


VARIANT_NAMES = {
    'direct_concat': 'DirectConcat',
    'gat': 'GAT',
    'graphsage': 'GraphSAGE',
    'jknet': 'JKNet',
    'no_self_feature': 'NoSelfFeature',
}
CONCURRENCY_GATE_PATH = (
    PROJECT_ROOT / 'training_process_data' / 'v1' / 'model'
    / 'ablation_training_status' / 'concurrency_gate.json'
)


def _pid_is_running(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _claim_training_slot():
    """Wait for the first batch, then cap later ablation training at six."""
    if not CONCURRENCY_GATE_PATH.is_file():
        return None
    gate = json.loads(CONCURRENCY_GATE_PATH.read_text(encoding='utf-8'))
    while any(_pid_is_running(pid) for pid in gate.get('initial_batch_pids', [])):
        time.sleep(15)

    slot_root = CONCURRENCY_GATE_PATH.parent / 'worker_slots'
    slot_root.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(int(gate.get('max_workers_after_initial_batch', 6))):
            slot_path = slot_root / f'slot{index}.lock'
            try:
                descriptor = os.open(str(slot_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    owner_pid = int(slot_path.read_text(encoding='ascii').strip())
                except (OSError, ValueError):
                    owner_pid = None
                if owner_pid is not None and _pid_is_running(owner_pid):
                    continue
                try:
                    slot_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(descriptor, 'w', encoding='ascii') as handle:
                handle.write(str(os.getpid()))
            return slot_path
        time.sleep(15)


def build_config(encoder_name, seed=0):
    """Copy the ISACR baseline and change only the named ablation contract."""
    if encoder_name not in VARIANT_NAMES:
        raise ValueError(f'Unknown ablation encoder: {encoder_name}')

    config = build_isacr_training_config('ISACR_self_elimination', seed)
    variant_name = VARIANT_NAMES[encoder_name]
    config['general']['algorithm_name'] = variant_name
    config['general']['random_seed'] = seed

    agent = config['agent']
    environment = config['environment']
    # Preserve the baseline's routing and computation settings.
    if encoder_name == 'no_self_feature':
        # Keep iSatCR shift aggregation and configured fault filling. Remove only the
        # recipient-specific self-feature elimination mechanism.
        environment['disable_self_feature_elimination'] = True
    else:
        # Alternative aggregators consume only currently transferable graph
        # data. Raw neighbor node features are not duplicated in the Q input.
        agent.update({
            'mode': 'GNN_DQN',
            'neighbors_dim': 0,
            'current_dim': 5,
            'in_channels': 6,
            'hidden_channels': 15,
            'edge_dim': 2,
            'gnn_dim': 15,
            'gnn_factory': make_encoder_factory(encoder_name),
        })
        environment.update({
            'gnn_information_hops': 3,
            'gnn_include_self': True,
            'gnn_node_feature_dim': 6,
            'historical_self_feature_elimination': False,
        })

    if encoder_name == 'graphsage':
        agent['hidden_channels'] = 5
        agent['gnn_dim'] = 5

    learning_rate = str(agent['learning_rate']).replace('.', 'p')
    gamma = str(agent['gamma']).replace('.', 'p')
    tag = f'seed{seed}_lr{learning_rate}_gamma{gamma}_repeat{agent["repeat"]}'
    seed_dir = Path('training_process_data') / 'v1' / 'model' / variant_name / f'seed{seed}'
    model_name = f'{variant_name}_seed{seed}'
    if encoder_name == 'no_self_feature':
        model_name += '.pth'
    agent['model_path'] = f'./{seed_dir.as_posix()}/{model_name}'
    environment['save_training_data'] = (
        f'v1/model/{variant_name}/seed{seed}/train_{variant_name}_{tag}.txt'
    )
    return config


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if callable(value):
        return f'{value.__module__}.{value.__qualname__}'
    return value


def save_config_snapshot(config):
    model_path = Path(config['agent']['model_path'])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    seed_dir = model_path.resolve().parent
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / 'training_config.json').write_text(
        json.dumps(_json_ready(config), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def run_variant(encoder_name, seed):
    config = build_config(encoder_name, seed)
    save_config_snapshot(config)
    run_training(config)


def main_for_encoder(encoder_name):
    from training_runner import prepare_reproducible_process

    prepare_reproducible_process()
    parser = argparse.ArgumentParser(description=f'Train {VARIANT_NAMES[encoder_name]}.')
    parser.add_argument('--seed', type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()
    selected_seeds = (args.seed,) if args.seed is not None else TRAINING_SEEDS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    slot_path = _claim_training_slot()
    try:
        for seed in selected_seeds:
            run_variant(encoder_name, seed)
    finally:
        if slot_path is not None:
            slot_path.unlink(missing_ok=True)
