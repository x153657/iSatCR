import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch

from training_runner import run_training, prepare_reproducible_process


CONFIG = {
    'general': {
        'phase': 'train',
        'random_seed': 0,
        'scenario_seed': 2020,
        'metric_checkpoint_names': True,
        'algorithm_name': 'ISACR_fault_self_elimination',
        'model_saving_enabled': True,
        'training_reward_save_start_step': 900,
        'epsilon': 0.9,
        'epsilon_decay': 0.995,
        'min_epsilon': 0.02,
        'begin_time': '2023-10-01 00:00:00',
        'time_stride': 1,
        'rounds': 1,
        'skip_time': (60, 0),
        'duration': 1000,
        'warmup_steps': 50,
        'reward_factors': (1, 0.05, 1, 0.1, 0.25),
        'print_cycle': 1,
        'show_test_summary': False,
        'select_mode': 1,
    },
    'agent': {
        'mode': 'New_DQN',
        'neighbors_dim': 72,
        'edges_dim': 4,
        'distance_dim': 4,
        'mission_dim': 6,
        'current_dim': 5,
        'action_dim': 5,
        'hidden_dim': 256,
        'buffer_length': 200000,
        'batch_size': 1024,
        'gamma': 0.99,
        'q_mask': -1,
        'activation': 'LeakyRelu',
        'hidden_layers': 2,
        'dueling': True,
        'shuffle': True,
        'learning_rate': 0.001,
        'repeat': 10,
        'use_huber_loss': False,
        'huber_delta': 1.0,
        'use_gradient_clipping': False,
        'max_grad_norm': 5.0,
        'experience_update': False,
        'update_cycle': 100,
        'initial_model_path': None,
        'model_path': './training_process_data/v1/model/ISACR_self_elimination/seed0/ISACR_fault_self_elimination_seed0_fill1p125_lr0p001_gamma0p99_repeat10.pth',
    },
    'environment': {
        'mission_possibility': [0.5, 0.5],
        'poisson_rate_s': 38.78,
        'poisson_rate_g': 19.39,
        'id': 'load_140',
        'target_load_tasks_per_second': 140,
        'packet_frequency': 1,
        'computing_demand_factor': [150, 250],
        'computing_demand_factor_2': [300, 500],
        'size_after_computing_factor': [0.0909, 0.1111],
        'size_after_computing_1': 40e3,
        'mean_interval_time': 30,
        'memory': 8e9,
        'computing_ability': 50e9,
        'transmission_rate': 1.2e9,
        'downlink_rate': 3e9,
        'downstream_delays': 0.0016667,
        'packet_size_range': [200e6, 600e6],
        'state_update_period': 0.1,
        'heartbeat_timeout': 0.15,
        'visualize': False,
        'print_info': False,
        'del_cycle': 30,
        'save_log': False,
        'show_detail': False,
        'random_edges_del': 0,
        'random_nodes_del': 0,
        'elevation_angle': 45,
        'update_cycle': 10,
        'pole': False,
        'tle_filepath': './Satellite_Data/wallker_tles.txt',
        'SOD_file_path': './Ground_Data/11_ground_stations.txt',
        'save_training_data': 'v1/model/ISACR_self_elimination/seed0/train_ISACR_fault_self_elimination_seed0_fill1p125_lr0p001_gamma0p99_repeat10.txt',
        'world_path': './ne_data/ne_50m_admin_0_countries_lakes.dbf',
        'markov_mode': True,
        'node_p': 0,
        'node_q': 0,
        'edge_p': 0.03,
        'edge_q': 0.97,
        'fault_compute_double_mean': True,
        'fault_fill_multiplier': 1.5,
        'uncomputed_reach_penalty': 0.0,
        'require_computed_before_downlink': False,
        # Send a recipient-specific 2-hop block and remove that recipient's
        # historical 1-hop contribution before the 3-hop aggregation.
        'historical_self_feature_elimination': True,
    },
}


FILL_MULTIPLIERS = (1.0, 1.125, 1.25, 1.375, 1.5)
TRAINING_SEEDS = (0, 1, 2, 3, 4)
TRAINING_WORKERS = len(TRAINING_SEEDS)


def multiplier_config(multiplier, seed):
    config = copy.deepcopy(CONFIG)
    tag = str(multiplier).replace('.', 'p')
    config['general']['random_seed'] = seed
    output_dir = f'ISACR_self_elimination/seed{seed}'
    config['environment']['fault_fill_multiplier'] = multiplier
    config['agent']['model_path'] = (
        f'./training_process_data/v1/model/{output_dir}/'
        f'ISACR_fault_self_elimination_seed{seed}_fill{tag}_lr0p001_gamma0p99_repeat10.pth'
    )
    config['environment']['save_training_data'] = (
        f'v1/model/{output_dir}/'
        f'train_ISACR_fault_self_elimination_seed{seed}_fill{tag}_lr0p001_gamma0p99_repeat10.txt'
    )
    return config


def run_training_seed(training_seed):
    completed = []
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for fill_multiplier in FILL_MULTIPLIERS:
        config = multiplier_config(fill_multiplier, training_seed)
        model_path = Path(config['agent']['model_path'])
        console_log_path = model_path.parent / (
            f'train_ISACR_fault_self_elimination_seed{training_seed}_console.txt'
        )
        console_log_path.parent.mkdir(parents=True, exist_ok=True)
        with console_log_path.open('w', encoding='utf-8', buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                run_training(config)
        completed.append((fill_multiplier, str(console_log_path)))
    return training_seed, completed


if __name__ == '__main__':
    prepare_reproducible_process()
    with ProcessPoolExecutor(
        max_workers=min(TRAINING_WORKERS, len(TRAINING_SEEDS))
    ) as executor:
        futures = {
            executor.submit(run_training_seed, training_seed): training_seed
            for training_seed in TRAINING_SEEDS
        }
        for future in as_completed(futures):
            seed, completed = future.result()
            print(
                f'Completed seed {seed}: '
                f'coefficients {[multiplier for multiplier, _ in completed]}',
                flush=True,
            )
