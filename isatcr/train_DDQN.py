import argparse
import copy

import torch

from training_runner import run_training as _run_training, prepare_reproducible_process


def run_training(config):
    from pathlib import Path
    import os
    import time
    hold = Path(__file__).resolve().parent / 'training_process_data/v1/model/four_5seed_training_status/DDQN_D3QN_WAIT_FOR_TESTS.lock'
    while hold.exists() and os.environ.get('SATELLITE_DDQN_D3QN_RELEASE') != '1':
        time.sleep(5)
    return _run_training(config)


CONFIG = {
    'general': {
        'phase': 'train',
        'random_seed': 0,
        'scenario_seed': 2020,
        'model_saving_enabled': True,
        'training_reward_save_start_step': 900,
        'metric_checkpoint_names': True,
        'algorithm_name': 'DDQN',
        'epsilon': 0.9,
        'epsilon_decay': 0.995,
        'min_epsilon': 0.02,
        'begin_time': '2023-10-01 00:00:00',
        'time_stride': 1,
        'rounds': 1,
        'skip_time': (60, 0),
        'duration': 1000,
        'warmup_steps': 151,
        'reward_factors': (1, 0.05, 1, 0.1, 0.25),
        'print_cycle': 1,
        'select_mode': 1,
    },
    'agent': {
        'mode': 'Pure_DQN',
        'neighbors_dim': 24,
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
        'dueling': False,
        'shuffle': True,
        'learning_rate': 0.001,
        'repeat': 10,
        # Optional robust-update stabilization.
        'use_huber_loss': False,
        'huber_delta': 1.0,
        'use_gradient_clipping': False,
        'max_grad_norm': 5.0,
        'experience_update': False,
        # One 1-second environment step performs 10 optimizer updates.
        # 100 optimizer updates (=10 seconds) synchronize Target Q.
        'update_cycle': 100,
        'model_path': './training_process_data/v1/model/DDQN/DDQN_seed42.pth',
    },
    'environment': {
        'mission_possibility': [0.5, 0.5],
        'poisson_rate_s': 29.15,
        'poisson_rate_g': 14.57,
        'id': 'load_160',
        'target_load_tasks_per_second': 160,
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
        'save_training_data': 'v1/model/DDQN/train_DDQN.txt',
        'world_path': './ne_data/ne_50m_admin_0_countries_lakes.dbf',
        'markov_mode': True,
        'node_p': 0,
        'node_q': 0,
        'edge_p': 0.18,
        'edge_q': 0.82,
        'fault_compute_double_mean': True,
        'fault_fill_multiplier': 2.0,
        'uncomputed_reach_penalty': 0.0,
        'require_computed_before_downlink': False,
    },
}


# Run all five training seeds for this training pass.
TRAINING_SEEDS = (0, 1, 2, 3, 4)


def seed_config(seed):
    config = copy.deepcopy(CONFIG)
    config['general']['random_seed'] = seed
    agent = config['agent']
    learning_rate = str(agent['learning_rate']).replace('.', 'p')
    gamma = str(agent['gamma']).replace('.', 'p')
    tag = (
        f'seed{seed}_lr{learning_rate}_gamma{gamma}'
        f'_repeat{agent["repeat"]}'
    )
    agent['model_path'] = f'./training_process_data/v1/model/DDQN/DDQN_seed{seed}.pth'
    config['environment']['save_training_data'] = (
        f'v1/model/DDQN/train_DDQN_{tag}.txt'
    )
    return config


if __name__ == '__main__':
    prepare_reproducible_process()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    selected_seeds = (args.seed,) if args.seed is not None else TRAINING_SEEDS
    for training_seed in selected_seeds:
        run_training(seed_config(training_seed))
