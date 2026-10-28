import os
import random
import sys
import copy
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

# Must be set before the first CUDA context/cuBLAS operation.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Base_Agents import DDQN_Agent, DQN_Agent, GNN_DDQN_Agent, PPO_Agent, ShuffleEx, ShuffleEx_gnn, cal_agent_dim
from RL_environment_for_computing import SatelliteEnv


def prepare_reproducible_process():
    """Fix Python hashing before startup, including spawned training workers."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        import subprocess

        os.environ['PYTHONHASHSEED'] = '0'
        # Keep the launcher alive until the child exits (also on Windows).
        raise SystemExit(subprocess.call(
            [sys.executable, '-B', '-u', *sys.argv], env=os.environ.copy(),
        ))


def configure_reproducible_training():
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError(
            'Training requires PYTHONHASHSEED=0 at Python startup; '
            'use the training entry point or prepare_reproducible_process().'
        )
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Fail explicitly instead of silently using a nondeterministic operator.
    torch.use_deterministic_algorithms(True)


def build_environment(config, agent, device, ppo_shuffle_func, *, scenario_seed=None,
                      print_cycle=None, save_training_data=None, experience_callback=None):
    general = config['general']
    environment = config['environment']
    mode = config['agent']['mode']
    return SatelliteEnv(
        mode=mode,
        select_mode=general['select_mode'],
        q_net=agent.online_net if agent else None,
        gnn=agent.online_gnn if 'GNN' in mode else None,
        ppo_shuffle_func=ppo_shuffle_func,
        epsilon=general['epsilon'],
        reward_factors=general['reward_factors'],
        uncomputed_reach_penalty=environment.get('uncomputed_reach_penalty', 0.0),
        device=device,
        mission_possibility=environment['mission_possibility'],
        poisson_rate_s=environment['poisson_rate_s'],
        poisson_rate_g=environment['poisson_rate_g'],
        packet_frequency=environment['packet_frequency'],
        computing_demand_factor=environment['computing_demand_factor'],
        computing_demand_factor_2=environment['computing_demand_factor_2'],
        size_after_computing_factor=environment['size_after_computing_factor'],
        size_after_computing_1=environment['size_after_computing_1'],
        begin_time=general['begin_time'],
        end_time=None,
        time_stride=general['time_stride'],
        tle_filepath=environment['tle_filepath'],
        SOD_file_path=environment['SOD_file_path'],
        world_path=environment['world_path'],
        mean_interval_time=environment['mean_interval_time'],
        memory=environment['memory'],
        computing_ability=environment['computing_ability'],
        transmission_rate=environment['transmission_rate'],
        downlink_rate=environment['downlink_rate'],
        downstream_delays=environment['downstream_delays'],
        packet_size_range=environment['packet_size_range'],
        state_update_period=environment['state_update_period'],
        heartbeat_timeout=environment.get('heartbeat_timeout', 0.15),
        print_cycle=print_cycle if print_cycle is not None else general['print_cycle'],
        del_cycle=environment['del_cycle'],
        visualize=environment['visualize'],
        print_info=environment['print_info'],
        show_detail=environment['show_detail'],
        save_log=environment['save_log'],
        random_edges_del=environment['random_edges_del'],
        random_nodes_del=environment['random_nodes_del'],
        update_cycle=environment['update_cycle'],
        save_training_data=(
            save_training_data
            if save_training_data is not None
            else environment['save_training_data']
        ),
        elevation_angle=environment['elevation_angle'],
        pole=environment['pole'],
        markov_mode=environment['markov_mode'],
        node_p=environment['node_p'],
        node_q=environment['node_q'],
        edge_p=environment['edge_p'],
        edge_q=environment['edge_q'],
        fault_compute_double_mean=environment.get('fault_compute_double_mean', True),
        fault_fill_multiplier=environment.get('fault_fill_multiplier', 2.0),
        historical_self_feature_elimination=environment.get('historical_self_feature_elimination', False),
        disable_self_feature_elimination=environment.get('disable_self_feature_elimination', False),
        require_computed_before_downlink=environment.get('require_computed_before_downlink', False),
        gnn_information_hops=environment.get('gnn_information_hops', 4),
        gnn_include_self=environment.get('gnn_include_self', False),
        gnn_node_feature_dim=environment.get('gnn_node_feature_dim', 3),
        tradition_replan_interval=environment.get('tradition_replan_interval'),
        tradition_use_genetic_search=environment.get('tradition_use_genetic_search', False),
        tradition_ga_population_size=environment.get('tradition_ga_population_size', 20),
        tradition_ga_generations=environment.get('tradition_ga_generations', 10),
        tradition_ga_path_candidates=environment.get('tradition_ga_path_candidates', 3),
        tradition_ga_mutation_rate=environment.get('tradition_ga_mutation_rate', 0.1),
        scenario_seed=(
            scenario_seed
            if scenario_seed is not None
            else general.get('scenario_seed')
        ),
        experience_callback=experience_callback,
        workload_trace_path=environment.get('workload_trace_path'),
        workload_trace_sha256=environment.get('workload_trace_sha256'),
        workload_trace_task_count=environment.get('workload_trace_task_count'),
        fault_trace_path=environment.get('fault_trace_path'),
        fault_trace_sha256=environment.get('fault_trace_sha256'),
        fault_trace_event_count=environment.get('fault_trace_event_count'),
    )


def best_model_path(model_path):
    path = Path(model_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_best{path.suffix}"))
    return str(path.parent / f"{path.name}_best")


def reward_best_model_path(model_path):
    path = Path(model_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_reward_best{path.suffix}"))
    return str(path.parent / f"{path.name}_reward_best")


def latest_model_path(model_path):
    """Return the single checkpoint path reserved for the final train state."""
    path = Path(model_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_latest{path.suffix}"))
    return str(path.parent / f"{path.name}_latest")


def find_latest_checkpoint(model_path):
    """Resolve a unique metric-labelled latest checkpoint or a legacy one."""
    path = Path(model_path)
    matches = [
        item for item in path.parent.glob(f'{path.stem}_latest_reward*{path.suffix}')
        if not item.name.endswith('.training.json')
    ]
    if matches:
        checkpoint = matches[0] if len(matches) == 1 else None
    else:
        checkpoint = Path(latest_model_path(model_path))
    if checkpoint is None:
        return None
    if path.suffix:
        return checkpoint if checkpoint.is_file() else None
    return checkpoint if any(
        all((checkpoint / name).is_file() for name in required)
        for required in (('actor.pth', 'critic.pth'), ('gnn.pth', 'qnet.pth'))
    ) else None


def _metric_filename_value(value, decimals):
    if value is None:
        return 'none'
    value = float(value)
    if not math.isfinite(value):
        return 'none'
    return f'{value:.{decimals}f}'.replace('-', 'm').replace('.', 'p')


def metric_checkpoint_path(config, category, metrics, selection_score=None):
    base_path = Path(config['agent']['model_path'])
    algorithm = config['general']['algorithm_name']
    seed = config['general']['random_seed']
    filename = (
        f'{algorithm}_seed{seed}_{category}'
        f'_reward{_metric_filename_value(metrics.get("average_reward"), 6)}'
        f'_delay{_metric_filename_value(metrics.get("average_delay"), 3)}'
        f'_loss{_metric_filename_value(metrics.get("packet_loss_rate"), 6)}'
        f'_hops{_metric_filename_value(metrics.get("average_hop_count"), 3)}'
    )
    if selection_score is not None:
        filename += f'_score{_metric_filename_value(selection_score, 6)}'
    return str(base_path.parent / f'{filename}{base_path.suffix}')


def _remove_checkpoint(path):
    path = Path(path)
    resolved_path = (
        path if path.is_absolute() else PROJECT_ROOT / path
    ).resolve()
    project_root = PROJECT_ROOT.resolve()
    if project_root not in resolved_path.parents:
        raise ValueError(f'Training output is outside project root: {resolved_path}')
    if resolved_path.is_file():
        resolved_path.unlink()
    elif resolved_path.is_dir():
        shutil.rmtree(resolved_path)


def training_input_hashes(config):
    paths = [PROJECT_ROOT / name for name in (
        'SatelliteNetworkSimulator_Computing.py', 'SatelliteNetworkSimulator_Beta.py',
        'RL_environment_for_computing.py', 'Base_Agents.py', 'training_runner.py',
        'SatelliteNetworkSimulation.py', 'Make_Satellite_Graph.py',
        'Read_Ground_Imformation.py',
    )]
    if 'GNN' in config['agent']['mode']:
        paths.extend(PROJECT_ROOT / name for name in (
            'ablation/aggregation_models.py', 'ablation/train_graphsage.py',
            'ablation/message_aggregation.py', 'ablation/training_entry.py',
            'replay_storage.py'))
    for key in ('tle_filepath', 'SOD_file_path'):
        paths.append((PROJECT_ROOT / config['environment'][key]).resolve())
    ability_path = paths[-2].with_name(paths[-2].stem + '_computing_ability.csv')
    if ability_path.is_file():
        paths.append(ability_path)
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def checkpoint_content_hashes(checkpoint_path):
    path = Path(checkpoint_path)
    files = [path] if path.is_file() else sorted(path.glob('*.pth'))
    return {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def save_training_checkpoint(agent, checkpoint_path, training_config):
    agent.save_model(checkpoint_path)
    metadata = {
        'config': training_config,
        'checkpoint_sha256': checkpoint_content_hashes(checkpoint_path),
    }
    Path(str(checkpoint_path) + '.training.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8',
    )


def load_checkpoint_training_config(checkpoint_path):
    metadata_path = Path(str(checkpoint_path) + '.training.json')
    if not metadata_path.is_file():
        raise ValueError(f'Missing checkpoint training metadata: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['checkpoint_sha256'] != checkpoint_content_hashes(checkpoint_path):
        raise ValueError(f'Checkpoint differs from training metadata: {checkpoint_path}')
    config = metadata['config']
    current_hashes = training_input_hashes(config)
    recorded_hashes = config.get('training_input_sha256', {})
    if recorded_hashes != current_hashes:
        changed = sorted(key for key in recorded_hashes.keys() | current_hashes.keys()
                         if recorded_hashes.get(key) != current_hashes.get(key))
        raise ValueError(f'Training/test source or resource mismatch: {changed}')
    return config


def save_replacing_checkpoint(agent, checkpoint_path, previous_path=None, *, training_config):
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    save_training_checkpoint(agent, checkpoint_path, training_config)
    if previous_path and Path(previous_path) != Path(checkpoint_path):
        _remove_checkpoint(previous_path)
        Path(str(previous_path) + '.training.json').unlink(missing_ok=True)
    return checkpoint_path


def reset_training_outputs(config):
    """Archive outputs from the same training seed before a fresh run."""
    agent = config['agent']
    output_paths = {
        Path(agent['model_path']),
        Path(best_model_path(agent['model_path'])),
        Path(reward_best_model_path(agent['model_path'])),
        Path(latest_model_path(agent['model_path'])),
    }
    if config['general'].get('metric_checkpoint_names'):
        model_parent = Path(agent['model_path']).parent
        search_parent = (
            model_parent
            if model_parent.is_absolute()
            else PROJECT_ROOT / model_parent
        )
        algorithm = config['general']['algorithm_name']
        seed = config['general']['random_seed']
        output_paths.update(search_parent.glob(f'{algorithm}_seed{seed}_*'))
    initial_model_path = agent.get('initial_model_path')
    initial_path = Path(initial_model_path) if initial_model_path else None
    project_root = PROJECT_ROOT.resolve()
    training_log = config['environment'].get('save_training_data')
    if training_log:
        output_paths.add(PROJECT_ROOT / 'training_process_data' / training_log)
    for output_path in tuple(output_paths):
        output_paths.add(Path(str(output_path) + '.training.json'))
    existing_outputs = []
    for output_path in output_paths:
        resolved_output = (
            output_path if output_path.is_absolute()
            else project_root / output_path
        ).resolve()
        if initial_path is not None:
            resolved_initial = (
                initial_path if initial_path.is_absolute()
                else project_root / initial_path
            ).resolve()
            if resolved_output == resolved_initial:
                continue
        if project_root not in resolved_output.parents:
            raise ValueError(f'Training output is outside project root: {resolved_output}')
        if resolved_output.exists():
            existing_outputs.append(resolved_output)
    if existing_outputs:
        archive_root = project_root / 'training_process_data' / 'v1' / 'model' / 'restart_archives'
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = Path(tempfile.mkdtemp(
            prefix=f"{config['general']['algorithm_name'][:8]}_seed{config['general']['random_seed']}_",
            dir=archive_root,
        ))
        for output_path in sorted(existing_outputs):
            target = archive / output_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output_path), str(target))
        print(f'Previous training outputs preserved: {archive}', flush=True)


def validation_metrics(validation_env, rewards, validation_duration, time_stride):
    statistics = validation_env.simulator.statics_datas
    reached = statistics['Reached_0'] + statistics['Reached_1']
    lost = (
        statistics['Lost_upload']
        + statistics['Lost_relay_0']
        + statistics['Lost_relay_1']
    )
    total_packets = reached + lost
    packet_loss_rate = lost / total_packets if total_packets else None
    average_delay = (
        (statistics['Total_delay_0'] + statistics['Total_delay_1']) / reached
        if reached else None
    )
    validation_steps = validation_duration / time_stride
    computing_ratio = (
        statistics['Is_computing'] / validation_env.num_nodes / validation_steps
        if validation_env.num_nodes and validation_steps else None
    )
    return {
        'average_reward': sum(rewards) / len(rewards),
        'computing_ratio': computing_ratio,
        'average_delay': average_delay,
        'packet_loss_rate': packet_loss_rate,
        'average_hop_count': (
            (statistics['Total_hops_0'] + statistics['Total_hops_1']) / reached
            if reached else None
        ),
        'generated_tasks': statistics['Total'],
        'unfinished_tasks': statistics['Total'] - reached - lost,
    }


def validate_agent(agent, config, device, ppo_shuffle_func):
    general = config['general']
    validation_duration = int(general['validation_duration'])
    time_stride = general['time_stride']
    validation_random_seed = int(
        general.get('validation_random_seed', general.get('validation_seed', general['random_seed']))
    )
    validation_scenario_seed = general.get(
        'validation_scenario_seed',
        general.get('validation_seed', general.get('scenario_seed')),
    )
    if validation_duration <= 0 or validation_duration % time_stride != 0:
        raise ValueError("validation_duration must be a positive multiple of time_stride")

    validation_config = copy.deepcopy(config)
    validation_config['general']['begin_time'] = general['begin_time']
    validation_config['general']['scenario_seed'] = validation_scenario_seed
    validation_config['general']['print_cycle'] = validation_duration + time_stride
    validation_config['environment']['print_info'] = False
    validation_config['environment']['save_log'] = False
    validation_config['environment']['save_training_data'] = None

    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modules = [getattr(agent, name, None) for name in ('online_net', 'critic_net', 'online_gnn')]
    modules = [module for module in modules if module is not None]
    training_states = [module.training for module in modules]
    policy = getattr(agent, 'online_net', None)
    previous_deterministic_eval = (
        getattr(policy, 'ppo_deterministic_eval', False)
        if policy is not None else False
    )

    validation_env = None
    try:
        random.seed(validation_random_seed)
        np.random.seed(validation_random_seed)
        torch.manual_seed(validation_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_random_seed)
        validation_env = build_environment(
            validation_config,
            agent,
            device,
            ppo_shuffle_func,
            scenario_seed=validation_scenario_seed,
            print_cycle=validation_duration + time_stride,
            save_training_data=None,
        )
        for module in modules:
            module.eval()
        if policy is not None and 'PPO' in config['agent']['mode']:
            policy.ppo_deterministic_eval = True

        validation_env.reset(general['begin_time'])
        rewards = []
        for _ in range(validation_duration // time_stride):
            validation_env.step(0)
            rewards.extend(validation_env.simulator.propagator.final_rewards)
        if not rewards:
            return None
        return validation_metrics(validation_env, rewards, validation_duration, time_stride)
    finally:
        if policy is not None and 'PPO' in config['agent']['mode']:
            policy.ppo_deterministic_eval = previous_deterministic_eval
        for module, training in zip(modules, training_states):
            module.train(training)
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _apply_validation_scenario(config, scenario):
    scenario_config = copy.deepcopy(config)
    environment_overrides = scenario.get('environment', {})
    general_overrides = scenario.get('general', {})
    if not environment_overrides and not general_overrides:
        environment_overrides = {
            key: value for key, value in scenario.items()
            if key in scenario_config['environment']
        }
        general_overrides = {
            key: value for key, value in scenario.items()
            if key in scenario_config['general']
        }
    scenario_config['environment'].update(environment_overrides)
    scenario_config['general'].update(general_overrides)
    return scenario_config


def validate_agent_scenarios(agent, config, device, ppo_shuffle_func):
    scenarios = config['general'].get('validation_scenarios')
    if not scenarios:
        return validate_agent(agent, config, device, ppo_shuffle_func)

    scenario_results = []
    abort_on_regression = bool(
        config['general'].get('validation_abort_on_incumbent_regression', False)
    )
    incumbent_by_name = {
        item['name']: (
            float(item['packet_loss_rate']),
            float(item['average_delay']),
            float(item['average_hop_count']),
        )
        for item in config['general'].get('validation_incumbent_metrics') or []
    }
    for index, scenario in enumerate(scenarios, start=1):
        scenario_config = _apply_validation_scenario(config, scenario)
        metrics = validate_agent(agent, scenario_config, device, ppo_shuffle_func)
        if metrics is None:
            return None
        scenario_name = scenario.get('name', f'scenario_{index}')
        scenario_results.append({
            'name': scenario_name,
            **metrics,
        })
        if abort_on_regression and scenario_name in incumbent_by_name:
            candidate = (
                metrics['packet_loss_rate'],
                metrics['average_delay'],
                metrics['average_hop_count'],
            )
            incumbent = incumbent_by_name[scenario_name]
            if any(
                value is None or not metric_no_worse(value, baseline)
                for value, baseline in zip(candidate, incumbent)
            ):
                raise RuntimeError(
                    f'Validation stopped at {scenario_name}: candidate metrics '
                    f'{candidate} regress from incumbent {incumbent}.'
                )

    def aggregate(metric, reducer):
        values = [result[metric] for result in scenario_results]
        if any(value is None for value in values):
            return None
        return reducer(values)

    return {
        'average_reward': aggregate('average_reward', np.mean),
        'computing_ratio': aggregate('computing_ratio', min),
        'average_delay': aggregate('average_delay', max),
        'packet_loss_rate': aggregate('packet_loss_rate', max),
        'average_hop_count': aggregate('average_hop_count', max),
        'generated_tasks': sum(result['generated_tasks'] for result in scenario_results),
        'unfinished_tasks': sum(result['unfinished_tasks'] for result in scenario_results),
        'scenario_results': scenario_results,
    }


def validation_detail_text(validation):
    details = validation.get('scenario_results')
    if not details:
        return ''
    return ' [' + '; '.join(
        f"{result['name']}: loss={result['packet_loss_rate']:.2%}, "
        f"delay={result['average_delay']:.3f}s, hops={result['average_hop_count']:.3f}"
        for result in details
    ) + ']'


def metric_no_worse(candidate, incumbent):
    return candidate < incumbent or math.isclose(
        candidate, incumbent, rel_tol=1e-12, abs_tol=1e-12
    )


def metric_strictly_better(candidate, incumbent):
    return candidate < incumbent and not math.isclose(
        candidate, incumbent, rel_tol=1e-12, abs_tol=1e-12
    )


def metrics_strictly_dominate(candidate, incumbent):
    """Return whether all tracked metrics improve or stay unchanged."""
    if incumbent is None:
        return True
    if candidate and isinstance(candidate[0], tuple):
        if not incumbent or not isinstance(incumbent[0], tuple):
            return False
        candidate_by_name = {item[0]: item[1:] for item in candidate}
        incumbent_by_name = {item[0]: item[1:] for item in incumbent}
        if candidate_by_name.keys() != incumbent_by_name.keys():
            return False
        candidate_values = [
            (candidate_by_name[name], incumbent_by_name[name])
            for name in sorted(candidate_by_name)
        ]
        return (
            all(
                metric_no_worse(candidate_metric, incumbent_metric)
                for candidate_metrics, incumbent_metrics in candidate_values
                for candidate_metric, incumbent_metric in zip(
                    candidate_metrics, incumbent_metrics
                )
            )
            and any(
                metric_strictly_better(candidate_metric, incumbent_metric)
                for candidate_metrics, incumbent_metrics in candidate_values
                for candidate_metric, incumbent_metric in zip(
                    candidate_metrics, incumbent_metrics
                )
            )
        )
    return (
        all(metric_no_worse(candidate[index], incumbent[index]) for index in range(3))
        and any(
            metric_strictly_better(candidate[index], incumbent[index])
            for index in range(3)
        )
    )


def validation_metric_key(validation):
    details = validation.get('scenario_results')
    if not details:
        return (
            validation['packet_loss_rate'],
            validation['average_delay'],
            validation['average_hop_count'],
        )
    return tuple(
        (
            result['name'],
            result['packet_loss_rate'],
            result['average_delay'],
            result['average_hop_count'],
        )
        for result in details
    )


def metric_key_selection_score(metric_key, max_average_delay, max_packet_loss_rate,
                               max_average_hop_count):
    """Return the normalized weighted validation score; lower is better."""
    if metric_key is None:
        return None
    values = metric_key if metric_key and isinstance(metric_key[0], tuple) else (metric_key,)
    scores = []
    for item in values:
        if isinstance(item[0], str):
            loss, delay, hops = item[1:]
        else:
            loss, delay, hops = item
        scores.append(
            0.6 * delay / max_average_delay
            + 0.3 * loss / max_packet_loss_rate
            + 0.1 * hops / max_average_hop_count
        )
    return float(np.mean(scores))


def validation_selection_score(validation, max_average_delay, max_packet_loss_rate,
                               max_average_hop_count):
    return metric_key_selection_score(
        validation_metric_key(validation),
        max_average_delay,
        max_packet_loss_rate,
        max_average_hop_count,
    )


def validation_config_signature(config):
    general = config['general']
    environment = {
        key: value for key, value in config['environment'].items()
        if key not in {
            'visualize', 'print_info', 'save_log', 'show_detail',
            'save_training_data',
        }
    }
    agent_fields = (
        'mode', 'neighbors_dim', 'edges_dim', 'distance_dim', 'mission_dim',
        'current_dim', 'action_dim', 'hidden_dim', 'q_mask', 'activation',
        'hidden_layers', 'dueling', 'shuffle',
    )
    payload = {
        'general': {
            key: general.get(key) for key in (
                'begin_time', 'time_stride', 'validation_duration',
                'validation_random_seed', 'validation_scenario_seed',
                'select_mode',
            )
        },
        'validation_scenarios': general.get('validation_scenarios'),
        'environment': environment,
        'agent': {key: config['agent'].get(key) for key in agent_fields},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def configured_validation_incumbent(config):
    general = config['general']
    metrics = general.get('validation_incumbent_metrics')
    if metrics is None:
        return None
    model_path = Path(general['validation_incumbent_model_path'])
    expected_model_hash = general['validation_incumbent_model_sha256'].lower()
    actual_model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_model_hash != expected_model_hash:
        raise ValueError('Configured validation incumbent model hash does not match')
    expected_signature = general['validation_incumbent_config_sha256'].lower()
    if validation_config_signature(config) != expected_signature:
        raise ValueError('Configured validation incumbent configuration does not match')
    expected_names = [
        scenario.get('name', f'scenario_{index}')
        for index, scenario in enumerate(general.get('validation_scenarios') or [], start=1)
    ]
    metric_names = [item['name'] for item in metrics]
    if metric_names != expected_names:
        raise ValueError('Configured validation incumbent scenarios do not match')
    return tuple(
        (
            item['name'],
            float(item['packet_loss_rate']),
            float(item['average_delay']),
            float(item['average_hop_count']),
        )
        for item in metrics
    )


def warmup_training_environment(env, steps, epsilon):
    """Build live queues without learning, then start a fresh reporting window."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError('warmup_steps must be a non-negative integer')
    if not steps:
        return
    env.print_and_save(f'Warmup started: {steps} steps, fixed epsilon={epsilon}, no optimizer updates')
    propagator = env.simulator.propagator
    callback = propagator.experience_callback
    propagator.experience_callback = None
    env.warmup_active = True
    try:
        for _ in range(steps):
            env.step(epsilon)
    finally:
        propagator.experience_callback = callback
        env.warmup_active = False
    propagator.experiences = []
    propagator.final_rewards = []
    env.rewards = []
    env.statics = [copy.deepcopy(env.simulator.statics_datas)]
    env.iteration_counter = 0
    env.last_average_ending_reward = None
    env.last_interval_metrics = None
    env.reporting_step_offset = env.step_num
    env.print_and_save(
        f'Warmup completed: physical_step={env.step_num}, '
        f'simulation_time={env.simulator.env.now}, queues retained; formal training starts at step 1'
    )


def run_training(config):
    os.chdir(PROJECT_ROOT)

    training_config = None
    if config['general']['phase'] == 'train':
        configure_reproducible_training()
        training_config = copy.deepcopy(config)
        factory = training_config['agent'].get('gnn_factory')
        if callable(factory):
            training_config['agent']['gnn_factory'] = f'{factory.__module__}.{factory.__qualname__}'
        training_config['reproducibility'] = {
            'python_hash_seed': '0',
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'deterministic_algorithms': True,
            'torch_num_threads': torch.get_num_threads(),
            'torch_version': str(torch.__version__),
            'cuda_version': torch.version.cuda,
        }
        training_config['training_input_sha256'] = training_input_hashes(config)

    random.seed(config['general']['random_seed'])
    torch.manual_seed(config['general']['random_seed'])
    np.random.seed(config['general']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['general']['random_seed'])

    phase = config['general']['phase']
    training_log = config['environment'].get('save_training_data')
    if phase == 'train' and training_log:
        reset_training_outputs(config)
        training_log_path = PROJECT_ROOT / 'training_process_data' / training_log
        training_log_path.parent.mkdir(parents=True, exist_ok=True)
        training_log_path.write_text('', encoding='utf-8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = config['agent']['mode']
    ppo_shuffle_func = None

    if mode in ['Pure_DQN', 'New_DQN', 'Pure_PPO', 'New_PPO', 'Weak_DQN', 'GNN_DQN']:
        state_dim, action_dim, state_mask = cal_agent_dim(
            neighbors_dim=config['agent']['neighbors_dim'],
            edges_dim=config['agent']['edges_dim'],
            distance_dim=config['agent']['distance_dim'],
            mission_dim=config['agent']['mission_dim'],
            current_dim=config['agent']['current_dim'],
            action_dim=config['agent']['action_dim'],
        )
        if 'GNN' in mode:
            agent = GNN_DDQN_Agent(
                in_channels=config['agent']['in_channels'],
                hidden_channels=config['agent']['hidden_channels'],
                edge_dim=config['agent']['edge_dim'],
                gnn_dim=config['agent']['gnn_dim'],
                other_dim=state_dim,
                hidden_dim=config['agent']['hidden_dim'],
                action_dim=action_dim,
                buffer_length=config['agent']['buffer_length'],
                batch_size=config['agent']['batch_size'],
                gamma=config['agent']['gamma'],
                device=device,
                q_mask=config['agent']['q_mask'],
                activation=config['agent']['activation'],
                hidden_layers=config['agent']['hidden_layers'],
                dueling=config['agent']['dueling'],
                learning_rate=config['agent']['learning_rate'],
                repeat=config['agent']['repeat'],
                shuffle_func=ShuffleEx_gnn(state_mask).shuffle if config['agent']['shuffle'] else None,
                gnn_factory=config['agent'].get('gnn_factory'),
                target_update_interval=config['agent']['update_cycle'],
                max_grad_norm=(config['agent'].get('max_grad_norm')
                               if config['agent'].get('use_gradient_clipping', True)
                               else None),
            )
            if getattr(agent.online_gnn, 'aligned_graphsage_policy', False):
                agent.shuffle_func = None
                if config['agent']['shuffle']:
                    ppo_shuffle_func = agent.online_gnn.shuffle_for_policy
        else:
            if 'DQN' in mode:
                Agent = DQN_Agent if 'Weak' in mode else DDQN_Agent
            else:
                Agent = PPO_Agent
            shuffle_ex = ShuffleEx(state_mask) if config['agent']['shuffle'] else None
            ppo_options = {
                key: config['agent'][key]
                for key in ('rollout_size', 'gae_lambda', 'clip_ratio', 'entropy_coef',
                            'max_grad_norm', 'target_kl', 'use_gradient_clipping')
                if key in config['agent']
            } if 'PPO' in mode else {}
            dqn_options = {
                key: config['agent'][key]
                for key in (
                    'learning_starts', 'train_freq', 'initial_updates',
                )
                if key in config['agent']
            } if 'DQN' in mode else {}
            if 'DQN' in mode:
                if 'huber_delta' in config['agent']:
                    dqn_options['huber_delta'] = (
                        config['agent']['huber_delta']
                        if config['agent'].get('use_huber_loss', True)
                        else None
                    )
                if 'max_grad_norm' in config['agent']:
                    dqn_options['max_grad_norm'] = (
                        config['agent']['max_grad_norm']
                        if config['agent'].get('use_gradient_clipping', True)
                        else None
                    )
            if Agent is DDQN_Agent:
                dqn_options['target_update_interval'] = config['agent']['update_cycle']
            agent = Agent(
                state_dim=state_dim,
                hidden_dim=config['agent']['hidden_dim'],
                action_dim=action_dim,
                buffer_length=config['agent']['buffer_length'],
                batch_size=config['agent']['batch_size'],
                gamma=config['agent']['gamma'],
                device=device,
                q_mask=config['agent']['q_mask'],
                activation=config['agent']['activation'],
                hidden_layers=config['agent']['hidden_layers'],
                dueling=config['agent']['dueling'],
                learning_rate=config['agent']['learning_rate'],
                repeat=config['agent']['repeat'],
                # Neighbor permutations are applied before action selection
                # and stored with each transition.  Replay must not apply a
                # second, current-state-only permutation.
                shuffle_func=None,
                **ppo_options,
                **dqn_options,
            )
            if shuffle_ex:
                ppo_shuffle_func = shuffle_ex.shuffle_for_policy
            initial_model_path = config['agent'].get('initial_model_path')
            if phase == 'train' and initial_model_path:
                agent.load_model(initial_model_path)
                if 'DQN' in mode:
                    if hasattr(agent, 'target_net'):
                        agent.target_net.load_state_dict(agent.online_net.state_dict())
                    if hasattr(agent, 'target_gnn') and hasattr(agent, 'online_gnn'):
                        agent.target_gnn.load_state_dict(agent.online_gnn.state_dict())
                    if hasattr(agent, '_updates_since_target'):
                        agent._updates_since_target = 0
        if phase != 'train':
            agent.load_model(config['agent']['model_path'])
    else:
        agent = None

    if phase != 'train':
        # Start evaluation RNG streams after model initialization and loading.
        random.seed(config['general']['random_seed'])
        np.random.seed(config['general']['random_seed'])
        torch.manual_seed(config['general']['random_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['general']['random_seed'])
        if agent is not None and 'PPO' in mode:
            agent.online_net.ppo_deterministic_eval = True

    optimizer_updates_since_report = 0
    optimizer_updates_total = 0
    use_experience_updates = (
        phase == 'train' and agent is not None
        and config['agent'].get('experience_update', False)
        and hasattr(agent, 'observe_experience')
    )

    def experience_callback(experience):
        nonlocal optimizer_updates_since_report, optimizer_updates_total
        stats = agent.observe_experience(experience)
        if stats is not None:
            optimizer_steps = int(stats.get('optimizer_steps', 0))
            optimizer_updates_since_report += optimizer_steps
            optimizer_updates_total += optimizer_steps

    if phase == 'train' and agent is not None:
        Path(config['agent']['model_path']).parent.mkdir(parents=True, exist_ok=True)

    env = build_environment(
        config,
        agent,
        device,
        ppo_shuffle_func,
        experience_callback=experience_callback if use_experience_updates else None,
    )

    begin_time = config['general']['begin_time']
    time_stride = config['general']['time_stride']
    rounds = config['general']['rounds']
    skip_time = config['general']['skip_time']
    duration = config['general']['duration']
    epsilon = config['general']['epsilon']
    min_epsilon = config['general']['min_epsilon']
    epsilon_decay = config['general']['epsilon_decay']
    validation_enabled = bool(config['general'].get('validation_enabled', False))
    validation_on_training_reward_best = bool(
        config['general'].get('validation_on_training_reward_best', False)
    )
    metric_checkpoint_names = bool(
        config['general'].get('metric_checkpoint_names', False)
    )
    model_saving_enabled = bool(
        config['general'].get('model_saving_enabled', True)
    )
    save_validation_models = bool(
        config['general'].get('save_validation_models', True)
    )
    best_metric_key = (
        configured_validation_incumbent(config) if validation_enabled else None
    )
    if best_metric_key is not None:
        env.print_and_save("Using verified configured validation incumbent.")
    validation_update_interval = int(config['general'].get('validation_update_interval', 480))
    validation_start_optimizer_updates = int(
        config['general'].get(
            'validation_start_optimizer_updates',
            validation_update_interval,
        )
    )
    next_validation_optimizer_steps = validation_start_optimizer_updates
    validation_step_interval = int(config['general'].get('validation_step_interval', 0))
    validation_start_step = config['general'].get('validation_start_step')
    use_simulation_step_validation = (
        validation_start_step is not None and validation_step_interval > 0
    )
    next_validation_simulation_step = (
        int(validation_start_step) if use_simulation_step_validation else None
    )
    best_model_save_start_step = int(
        config['general'].get('best_model_save_start_step', 0)
    )
    training_reward_save_start_step = int(
        config['general'].get('training_reward_save_start_step', 900)
    )
    max_average_delay = float(config['general'].get('validation_max_average_delay', 3.0))
    max_packet_loss_rate = float(config['general'].get('validation_max_packet_loss_rate', 0.01))
    max_average_hop_count = float(
        config['general'].get('validation_max_average_hop_count', float('inf'))
    )
    best_validation_score = metric_key_selection_score(
        best_metric_key,
        max_average_delay,
        max_packet_loss_rate,
        max_average_hop_count,
    )
    best_training_reward = None
    best_validation_reward = None
    checkpoint_paths = {}
    latest_training_metrics = {}
    save_reward_best = (
        model_saving_enabled and phase == 'train' and agent is not None
    )
    if best_metric_key is not None and config['general'].get('validate_initial_model', False):
        raise ValueError(
            'Use either validate_initial_model or validation_incumbent_metrics, not both'
        )
    validation_function = (
        validate_agent_scenarios
        if config['general'].get('validation_scenarios')
        else validate_agent
    )
    training_scenarios = config['general'].get('training_scenarios') or []
    print_cycle_iterations = max(int(config['general']['print_cycle'] / time_stride), 1)

    if phase != 'train':
        epsilon = 0

    if (
        phase == 'train'
        and agent is not None
        and 'DQN' in mode
        and validation_enabled
        and config['general'].get('validate_initial_model', False)
    ):
        initial_validation = validation_function(agent, config, device, ppo_shuffle_func)
        if initial_validation is not None:
            initial_delay = initial_validation['average_delay']
            initial_loss = initial_validation['packet_loss_rate']
            initial_hops = initial_validation['average_hop_count']
            if (
                model_saving_enabled
                and save_validation_models
                and best_model_save_start_step <= 0
                and initial_delay is not None
                and initial_loss is not None
                and initial_hops is not None
            ):
                best_metric_key = validation_metric_key(initial_validation)
                best_validation_score = validation_selection_score(
                    initial_validation,
                    max_average_delay,
                    max_packet_loss_rate,
                    max_average_hop_count,
                )
            env.print_and_save(
                f"Initial validation: reward={initial_validation['average_reward']:.6f}, "
                f"computing={initial_validation['computing_ratio']:.2%}, "
                f"delay={initial_delay:.3f}s, packet_loss={initial_loss:.2%}, "
                f"hops={initial_hops}, "
                f"unfinished={initial_validation['unfinished_tasks']}/{initial_validation['generated_tasks']}"
                f"{validation_detail_text(initial_validation)}"
            )
            if (
                best_model_save_start_step <= 0
                and best_validation_score is not None
            ):
                initial_best_path = best_model_path(config['agent']['model_path'])
                Path(initial_best_path).parent.mkdir(parents=True, exist_ok=True)
                save_training_checkpoint(agent, initial_best_path, training_config)
                env.print_and_save(
                    f"Initial model saved as best baseline: {initial_best_path} "
                    f"(selection_score={best_validation_score:.6f})"
                )

    for round_index in range(rounds):
        if training_scenarios:
            scenario = training_scenarios[round_index % len(training_scenarios)]
            scenario_config = _apply_validation_scenario(config, scenario)
            env.apply_scenario(scenario_config['environment'])
            env.print_and_save(
                f"Training scenario {scenario.get('name', round_index + 1)}"
            )
        env.reset(begin_time)
        if phase == 'train':
            warmup_training_environment(
                env, config['general'].get('warmup_steps', 0), epsilon,
            )
        for t in range(int(duration / time_stride)):
            experiences = env.step(epsilon)
            if phase == 'train' and agent:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                if use_experience_updates:
                    pass
                elif 'PPO' in mode:
                    stats = agent.update(experiences)
                    if stats is not None:
                        optimizer_steps = int(stats.get('optimizer_steps', 0))
                        optimizer_updates_since_report += optimizer_steps
                        optimizer_updates_total += optimizer_steps
                        env.print_and_save(
                            'PPO rollout: '
                            f"samples={stats.get('samples', 0)}, "
                            f"policy_version={stats.get('policy_version', 'n/a')}, "
                            f"bootstrap_cuts={stats.get('bootstrap_cuts', 0)}, "
                            f"discarded_stale={stats.get('discarded_stale', 0)}"
                        )
                else:
                    optimizer_steps = int(agent.update(experiences) or 0)
                    optimizer_updates_since_report += optimizer_steps
                    optimizer_updates_total += optimizer_steps
                if (t + 1) % print_cycle_iterations == 0:
                    target_updates = getattr(agent, 'target_update_events', None)
                    target_text = (
                        f", target Q syncs={target_updates}"
                        if target_updates is not None else ""
                    )
                    env.print_and_save(
                        f"Parameter updates in last {print_cycle_iterations} steps: "
                        f"{optimizer_updates_since_report} "
                        f"(total={optimizer_updates_total}{target_text})"
                    )
                    optimizer_updates_since_report = 0
            simulation_step = round_index * int(duration / time_stride) + t + 1
            interval_metrics = getattr(env, 'last_interval_metrics', None)
            training_reward = (
                interval_metrics.get('average_reward')
                if interval_metrics is not None
                else env.last_average_ending_reward
            )
            if interval_metrics is not None or training_reward is not None:
                latest_training_metrics = dict(interval_metrics or {})
                latest_training_metrics['average_reward'] = training_reward
            training_reward_improved = False
            if (
                save_reward_best
                and training_reward is not None
                and simulation_step >= training_reward_save_start_step
                and (
                    best_training_reward is None
                    or training_reward > best_training_reward
                )
            ):
                training_reward_improved = True
                best_training_reward = training_reward
                if metric_checkpoint_names:
                    checkpoint_metrics = dict(interval_metrics or {})
                    checkpoint_metrics['average_reward'] = training_reward
                    reward_best_path = metric_checkpoint_path(
                        config,
                        'train_reward_best',
                        checkpoint_metrics,
                    )
                    checkpoint_paths['train_reward_best'] = save_replacing_checkpoint(
                        agent,
                        reward_best_path,
                        checkpoint_paths.get('train_reward_best'),
                        training_config=training_config,
                    )
                else:
                    reward_best_path = reward_best_model_path(
                        config['agent']['model_path']
                    )
                    Path(reward_best_path).parent.mkdir(parents=True, exist_ok=True)
                    save_training_checkpoint(agent, reward_best_path, training_config)
                env.print_and_save(
                    f"Training reward-best model saved: {reward_best_path} "
                    f"(average_reward={training_reward:.6f}, "
                    f"simulation_step={simulation_step})"
                )
            env.last_average_ending_reward = None
            if hasattr(env, 'last_interval_metrics'):
                env.last_interval_metrics = None
            optimizer_steps_total = optimizer_updates_total
            if validation_on_training_reward_best:
                validation_due = (
                    validation_enabled
                    and training_reward_improved
                    and (
                        validation_start_step is None
                        or simulation_step >= int(validation_start_step)
                    )
                )
            elif use_simulation_step_validation:
                validation_due = (
                    validation_enabled
                    and validation_step_interval > 0
                    and simulation_step >= next_validation_simulation_step
                )
            else:
                validation_due = (
                    validation_enabled
                    and validation_update_interval > 0
                    and optimizer_steps_total >= next_validation_optimizer_steps
                    and (
                        validation_start_step is None
                        or simulation_step >= int(validation_start_step)
                    )
                )
            if validation_due:
                if validation_on_training_reward_best:
                    validation_label = (
                        f'Validation triggered by training reward best at '
                        f'{simulation_step}s'
                    )
                    env.print_and_save(
                        f'Starting validation after training reward improved to '
                        f'{training_reward:.6f} at simulation time {simulation_step}s'
                    )
                elif use_simulation_step_validation:
                    while next_validation_simulation_step <= simulation_step:
                        next_validation_simulation_step += validation_step_interval
                    validation_label = f"Validation simulation time {simulation_step}s"
                    env.print_and_save(f"Starting validation at simulation time {simulation_step}s")
                else:
                    while next_validation_optimizer_steps <= optimizer_steps_total:
                        next_validation_optimizer_steps += validation_update_interval
                    validation_label = f"Validation optimizer updates {optimizer_steps_total}"
                    env.print_and_save(
                        f"Starting validation at optimizer updates {optimizer_steps_total}"
                    )
                validation = validation_function(agent, config, device, ppo_shuffle_func)
                if validation is not None:
                    average_reward = validation['average_reward']
                    computing_ratio = validation['computing_ratio']
                    average_delay = validation['average_delay']
                    packet_loss_rate = validation['packet_loss_rate']
                    average_hop_count = validation['average_hop_count']
                    computing_label = (
                        f"computing={computing_ratio:.2%}"
                        if computing_ratio is not None else "computing=None"
                    )
                    delay_label = (
                        f"delay={average_delay:.3f}s"
                        if average_delay is not None else "delay=None"
                    )
                    loss_label = (
                        f"packet_loss={packet_loss_rate:.2%}"
                        if packet_loss_rate is not None else "packet_loss=None"
                    )
                    env.print_and_save(
                        f"{validation_label}: reward={average_reward:.6f}, "
                        f"{computing_label}, {delay_label}, {loss_label}, "
                        f"hops={validation['average_hop_count']}, "
                        f"unfinished={validation['unfinished_tasks']}/{validation['generated_tasks']}"
                        f"{validation_detail_text(validation)}"
                    )
                    candidate_metric_key = validation_metric_key(validation)
                    candidate_score = None
                    if all(
                        value is not None
                        for value in (
                            average_delay,
                            packet_loss_rate,
                            average_hop_count,
                        )
                    ):
                        candidate_score = validation_selection_score(
                            validation,
                            max_average_delay,
                            max_packet_loss_rate,
                            max_average_hop_count,
                        )
                    if not model_saving_enabled:
                        env.print_and_save(
                            'Model saving is disabled for this training run.'
                        )
                        continue
                    if metric_checkpoint_names:
                        if (
                            save_validation_models
                            and (
                            best_validation_reward is None
                            or average_reward > best_validation_reward
                            )
                        ):
                            best_validation_reward = average_reward
                            test_reward_path = metric_checkpoint_path(
                                config,
                                'test_reward_best',
                                validation,
                            )
                            checkpoint_paths['test_reward_best'] = save_replacing_checkpoint(
                                agent,
                                test_reward_path,
                                checkpoint_paths.get('test_reward_best'),
                                training_config=training_config,
                            )
                            env.print_and_save(
                                f'Test reward-best model saved: {test_reward_path} '
                                f'(reward={average_reward:.6f}, {delay_label}, '
                                f'{loss_label}, '
                                f'hops={average_hop_count})'
                            )
                        if (
                            save_validation_models
                            and candidate_score is not None
                            and (
                                best_validation_score is None
                                or candidate_score < best_validation_score
                            )
                        ):
                            best_metric_key = candidate_metric_key
                            best_validation_score = candidate_score
                            test_score_path = metric_checkpoint_path(
                                config,
                                'test_score_best',
                                validation,
                                selection_score=candidate_score,
                            )
                            checkpoint_paths['test_score_best'] = save_replacing_checkpoint(
                                agent,
                                test_score_path,
                                checkpoint_paths.get('test_score_best'),
                                training_config=training_config,
                            )
                            env.print_and_save(
                                f'Test score-best model saved: {test_score_path} '
                                f'(reward={average_reward:.6f}, delay={average_delay:.3f}s, '
                                f'packet_loss={packet_loss_rate:.2%}, '
                                f'hops={average_hop_count}, '
                                f'selection_score={candidate_score:.6f})'
                            )
                        continue
                    candidate_improves = (
                        save_validation_models
                        and simulation_step >= best_model_save_start_step
                        and candidate_score is not None
                        and (
                            best_validation_score is None
                            or candidate_score < best_validation_score
                        )
                    )
                    if candidate_improves:
                        best_metric_key = candidate_metric_key
                        best_validation_score = candidate_score
                        best_path = best_model_path(
                            config['agent']['model_path']
                        )
                        Path(best_path).parent.mkdir(parents=True, exist_ok=True)
                        save_training_checkpoint(agent, best_path, training_config)
                        env.print_and_save(
                            f"Best model saved: {best_path} "
                            f"(reward={average_reward:.6f}, computing={computing_ratio:.2%}, "
                            f"delay={average_delay:.3f}s, packet_loss={packet_loss_rate:.2%}, "
                            f"hops={average_hop_count}, selection_score={candidate_score:.6f})"
                        )
                    elif simulation_step < best_model_save_start_step:
                        env.print_and_save(
                            "Best model saving is disabled until environment step "
                            f"{best_model_save_start_step}."
                        )
                    else:
                        env.print_and_save(
                            "Best model not updated: weighted selection score did not improve."
                        )
        if phase == 'test':
            env.validate_fixed_trace_replay()
            if config['general'].get('show_test_summary', True):
                env.show_satellite_computing_time()
                env.show_network_transmission_occupation(duration)
                env.show_equivalent_transmission_data(duration)
                env.show_test_trace_fingerprints()
        begin_time = env.add_time_to_str(begin_time, skip_time)

    if phase == 'train' and model_saving_enabled and agent is not None:
        latest_path = (
            metric_checkpoint_path(config, 'latest', latest_training_metrics)
            if metric_checkpoint_names
            else latest_model_path(config['agent']['model_path'])
        )
        Path(latest_path).parent.mkdir(parents=True, exist_ok=True)
        save_training_checkpoint(agent, latest_path, training_config)
        env.print_and_save(f"Latest model saved: {latest_path}")
# BEGIN OPTIONAL_RUNTIME_ACCELERATION_V1
# Optional execution layer. Keep the original prefix unchanged so legacy model
# compatibility can verify that exact prefix, rather than bypass source checks.
import inspect
import textwrap
import threading
import gc
from collections import deque
from contextlib import contextmanager

ACCELERATION_ENABLED = True
ACCELERATION_CPU_POLICY = True
ACCELERATION_ROUTING = True
ACCELERATION_REPLAY = True
ACCELERATION_PPO_SAMPLED_ACTION = True
ACCELERATION_COMPACT_MESSAGES = False

_speed_lock = threading.RLock()
_speed_original_build_environment = build_environment
_speed_original_run_training = run_training
_speed_original_load_checkpoint = load_checkpoint_training_config


@contextmanager
def _speed_satellite_scope(simulation, gnn, satellite_class):
    with _speed_lock:
        original = simulation.Satellite_with_Computing
        owned_class = gnn is not None and 'satellite_class' in vars(gnn)
        previous_class = getattr(gnn, 'satellite_class', None)
        simulation.Satellite_with_Computing = satellite_class
        if gnn is not None:
            gnn.satellite_class = satellite_class
        try:
            yield
        finally:
            simulation.Satellite_with_Computing = original
            if gnn is not None:
                if owned_class:
                    gnn.satellite_class = previous_class
                else:
                    del gnn.satellite_class


def acceleration_settings():
    names = {'enabled': ('SATELLITE_ACCELERATION', ACCELERATION_ENABLED),
             'cpu_policy': ('SATELLITE_ACCELERATION_CPU', ACCELERATION_CPU_POLICY),
             'routing': ('SATELLITE_ACCELERATION_ROUTING', ACCELERATION_ROUTING),
             'replay': ('SATELLITE_ACCELERATION_REPLAY', ACCELERATION_REPLAY),
             'ppo_sampled_action': ('SATELLITE_ACCELERATION_PPO', ACCELERATION_PPO_SAMPLED_ACTION),
             'compact_messages': ('SATELLITE_ACCELERATION_MEMORY', ACCELERATION_COMPACT_MESSAGES)}
    result = {}
    for key, (variable, default) in names.items():
        value = os.environ.get(variable)
        if value is None:
            result[key] = default
        elif value.lower() in ('1', 'true', 'on'):
            result[key] = True
        elif value.lower() in ('0', 'false', 'off'):
            result[key] = False
        else:
            raise ValueError(f'{variable} must be 0/1, false/true, or off/on')
    return result


def runtime_acceleration_contract():
    return {'version': 1, 'settings': acceleration_settings(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class _SpeedAdjacency(dict):
    """Cache immutable outgoing rows when a remote record is replaced."""
    def __init__(self, owner, values):
        self.owner = owner
        self.frozen = {}
        for key, value in values.items():
            self[key] = value

    def freeze(self, key, value):
        neighbors, timestamp = value
        row = (tuple(neighbors), timestamp)
        propagator = self.owner.propagator
        if propagator is not None:
            rows = getattr(propagator, '_speed_adjacency_rows', None)
            if rows is None:
                rows = propagator._speed_adjacency_rows = {}
            previous = rows.get(key)
            if previous == row:
                row = previous
            else:
                rows[key] = row
        self.frozen[key] = row
        if hasattr(self, 'versions'):
            self.versions[self.node_ids[key]] = timestamp
            if self.key_order is not None and len(self.key_order) != len(self.frozen):
                self.key_order = None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.freeze(key, value)

    def prepare_versions(self):
        if hasattr(self, 'versions'):
            return
        propagator = self.owner.propagator
        if not hasattr(propagator, '_speed_adjacency_ids'):
            propagator._speed_adjacency_ids = {
                name: i for i, name in enumerate(propagator.satellites)}
        self.node_ids = propagator._speed_adjacency_ids
        self.versions = np.full(len(self.node_ids), -np.inf)
        for key, (_, timestamp) in self.frozen.items():
            self.versions[self.node_ids[key]] = timestamp
        self.key_order = None


class _SpeedAdjacencyPayload(dict):
    __slots__ = ('indices', 'key_order', 'timestamps')


def _speed_get_adjacency(self):
    return self.__dict__['adjacency_table']


def _speed_set_adjacency(self, value):
    self.__dict__['adjacency_table'] = _SpeedAdjacency(self, value)


def _speed_merge_adjacency(self, incoming):
    local = self.adjacency_table
    if isinstance(incoming, _SpeedAdjacencyPayload) and isinstance(local, _SpeedAdjacency):
        local.prepare_versions()
        changed = np.flatnonzero(incoming.timestamps > local.versions[incoming.indices])
        if not len(changed):
            return False
        # Retain the incoming insertion order, including equal-version behavior.
        keys = [incoming.key_order[i] for i in changed]
        incoming = {key: incoming[key] for key in keys}
    return self._speed_original_merge_adjacency(incoming)


def _speed_route_property(name):
    def get(self):
        data = self.__dict__
        if data.get('_speed_routes_dirty', False) and not data.get('_speed_routes_building', False):
            data['_speed_routes_dirty'] = False
            data['_speed_routes_building'] = True
            snapshot = data.get('_speed_route_inputs')
            live = None
            if snapshot is not None:
                # A timestamp-only update need not request a rebuild. Use the
                # inputs at the last original rebuild call, not at this read.
                live = (data['adjacency_table'], self.neighbors, self._neighbor_route_since)
                data['adjacency_table'], self.neighbors, self._neighbor_route_since = snapshot
            try:
                self._speed_eager_routes()
            except BaseException:
                data['_speed_routes_dirty'] = True
                raise
            finally:
                if live is not None:
                    data['adjacency_table'], self.neighbors, self._neighbor_route_since = live
                data['_speed_routes_building'] = False
        return data[name]

    def set(self, value):
        self.__dict__[name] = value

    return property(get, set)


def _speed_defer_routes(self):
    # Routing is pure derived state. Resolve before any read, without adding,
    # deleting or rescheduling simulation events or drawing random numbers.
    adjacency = getattr(self, 'adjacency_table', None)
    if adjacency is not None:
        if isinstance(adjacency, _SpeedAdjacency):
            adjacency.freeze(self.name, adjacency[self.name])
            frozen = adjacency.frozen.copy()
        else:
            frozen = {key: (tuple(neighbors), timestamp)
                      for key, (neighbors, timestamp) in adjacency.items()}
        self.__dict__['_speed_route_inputs'] = (
            frozen, list(self.neighbors), dict(self._neighbor_route_since))
    self.__dict__['_speed_routes_dirty'] = True


def _speed_adjacency_table_exchanger(self):
    if isinstance(self.adjacency_table, _SpeedAdjacency):
        adjacency = self.adjacency_table
        adjacency.prepare_versions()
        # Only the local row aliases a live neighbor list; remote rows are
        # replaced by update_adjacency_dict, never mutated in place.
        adjacency.freeze(self.name, adjacency[self.name])
        if adjacency.key_order is None:
            adjacency.key_order = tuple(adjacency.frozen)
            adjacency.indices = np.array([adjacency.node_ids[k] for k in adjacency.key_order])
        table = _SpeedAdjacencyPayload(adjacency.frozen)
        table.indices = adjacency.indices
        table.key_order = adjacency.key_order
        table.timestamps = adjacency.versions[adjacency.indices]
        for neighbor in self.neighbors:
            self.env.process(self.propagator._deliver_adjacency_table(self.name, neighbor, table))
        return
    # Delivery only reads these records; receivers already copy neighbor lists.
    # Freeze once before scheduling, just as send_adjacency_table does, and keep
    # every delivery event in its original order. Reuse equal immutable rows
    # across outstanding messages instead of copying a list for each recipient.
    propagator = self.propagator
    rows = getattr(propagator, '_speed_adjacency_rows', None)
    if rows is None:
        rows = propagator._speed_adjacency_rows = {}
    table = {}
    for key, (neighbors, timestamp) in self.adjacency_table.items():
        row = (tuple(neighbors), timestamp)
        previous = rows.get(key)
        if previous == row:
            row = previous
        else:
            rows[key] = row
        table[key] = row
    for neighbor in self.neighbors:
        self.env.process(propagator._deliver_adjacency_table(self.name, neighbor, table))


def _speed_routing_table(self):
    routing_adjacency = self.update_adjacency_dict_for_bfs()
    adjacency = {node: sorted(value[0]) for node, value in routing_adjacency.items()}
    for neighbor in self._neighbor_route_since:
        adjacency[neighbor] = []
    for neighbor in self.neighbors:
        adjacency.setdefault(neighbor, [])
    cache_key = (self.name, tuple(self.neighbors), adjacency)
    cached = getattr(self, '_routing_cache', None)
    if cached is not None and cached[0] == cache_key:
        self.routing_tables = {node: (list(next_hops), hops)
                               for node, (next_hops, hops) in cached[1].items()}
        self.neighbor_hops = {node: dict(hops) for node, hops in cached[2].items()}
        return
    result = {}
    self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}
    for start in [self.name] + self.neighbors:
        if start == self.name:
            result = _speed_root_routes(adjacency, start)
        else:
            self.neighbor_hops[start] = _speed_neighbor_distances(adjacency, start)
    result[self.name] = ([self.name], 0)
    self.routing_tables = result
    self._routing_cache = (
        cache_key,
        {node: (list(next_hops), hops) for node, (next_hops, hops) in result.items()},
        {node: dict(hops) for node, hops in self.neighbor_hops.items()},
    )


def _speed_sampled_action_policy(original):
    source = textwrap.dedent(inspect.getsource(original))
    marker = 'action = dist.sample()'
    constructor = 'torch.distributions.Categorical(logits=main_output)'
    extraction = ('policy_action = action.item()\n'
                  '            log_probability = dist.log_prob(action).item()')
    if (source.count(marker) != 1 or source.count(constructor) != 1
            or source.count(extraction) != 1):
        raise ValueError('PPO policy implementation changed; review acceleration')
    # Retain CUDA normalization, multinomial sampling and log_prob exactly.
    # Distribution debug checks otherwise synchronize the GPU on every packet.
    source = source.replace(constructor,
        'torch.distributions.Categorical(logits=main_output, validate_args=False)')
    source = source.replace(marker, marker + '\n            dist._validate_args = False')
    # Finish both GPU results before one host read; retain sample/log_prob math.
    source = source.replace(extraction,
        'log_probability_tensor = dist.log_prob(action)\n'
        '            result = torch.stack((action.to(log_probability_tensor.dtype), '
        'log_probability_tensor), dim=-1).tolist()[0]\n'
        '            policy_action = int(result[0])\n'
        '            log_probability = result[1]')
    namespace = dict(inspect.unwrap(original).__globals__)
    exec(compile(source, __file__, 'exec'), namespace)
    return namespace['get_next_hop']


class IndexedReplay:
    def __init__(self, values=(), maxlen=200000):
        if maxlen <= 0:
            raise ValueError('Replay capacity must be positive')
        self.maxlen = maxlen
        self.values = []
        self.start = 0
        self.extend(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        size = len(self.values)
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError('Replay index out of range')
        physical_index = self.start + index
        if physical_index >= size:
            physical_index -= size
        return self.values[physical_index]

    def append(self, value):
        if len(self.values) < self.maxlen:
            self.values.append(value)
        else:
            self.values[self.start] = value
            self.start = (self.start + 1) % self.maxlen

    def extend(self, values):
        for value in values:
            self.append(value)

    def clear(self):
        self.values.clear()
        self.start = 0


def _speed_neighbor_distances(adjacency, start):
    distances = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        hops = distances[node] + 1
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = hops
                queue.append(neighbor)
    return distances


def _speed_root_routes(adjacency, start):
    result = {}
    queue = deque()
    for neighbor in adjacency[start]:
        if neighbor not in result:
            result[neighbor] = ([neighbor], 1)
            queue.append(neighbor)
        else:
            result[neighbor][0].append(neighbor)
    while queue:
        node = queue.popleft()
        first_hops, hops = result[node]
        first_hop = first_hops[0]
        for neighbor in adjacency[node]:
            if neighbor not in result:
                result[neighbor] = ([first_hop], hops + 1)
                queue.append(neighbor)
            elif result[neighbor][1] == hops + 1:
                result[neighbor][0].append(first_hop)
    return result


def _speed_build_cpu_environment(build, config, agent, device, shuffle, **kwargs):
    if 'GNN' in config['agent']['mode']:
        raise ValueError('This experiment only supports the four plain policy networks')
    if kwargs.get('experience_callback') is not None and 'PPO' not in config['agent']['mode']:
        raise ValueError('Updates inside a simulation step need separate synchronization')
    source = agent.online_net
    cpu_net = copy.deepcopy(source).cpu()
    is_ppo = 'PPO' in config['agent']['mode']
    from Base_Agents import QNetwork
    # Script only the shared plain Q network. Parameters share storage with
    # cpu_net, so the existing per-step weight synchronization remains valid.
    scripted_net = torch.jit.script(cpu_net) if type(cpu_net) is QNetwork and not is_ppo else None

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = cpu_net

        def forward(self, *args, **kwargs):
            if scripted_net is not None:
                state = args[0]
                mask = args[1] if len(args) > 1 else kwargs.get('valid_action_mask')
                if mask is not None:
                    mask = torch.as_tensor(mask, dtype=torch.bool, device=state.device)
                    return scripted_net(state, mask)
            result = self.network(*args, **kwargs)
            # Keep Categorical sampling and log-probability on the original
            # device, so PPO consumes the same random-number stream.
            return result.to(device) if is_ppo else result

    policy = Policy()
    proxy = copy.copy(agent)
    proxy.online_net = policy
    original_callback = kwargs.get('experience_callback')
    if original_callback is not None:
        def callback(experience):
            original_callback(experience)
            if getattr(source, 'ppo_policy_version', None) != getattr(policy, 'ppo_policy_version', None):
                sync()
        kwargs['experience_callback'] = callback
    env = build(config, proxy, torch.device('cpu'), shuffle, **kwargs)
    original_step = env.step
    versions = None
    env.cpu_inference_syncs = 0

    def sync():
        nonlocal versions
        tensors = tuple(source.parameters()) + tuple(source.buffers())
        current = tuple((id(value), value._version) for value in tensors)
        if current != versions:
            cpu_net.load_state_dict({key: value.detach().cpu()
                                     for key, value in source.state_dict().items()})
            versions = current
            env.cpu_inference_syncs += 1
        cpu_net.train(source.training)
        for name in ('ppo_policy_version', 'ppo_deterministic_eval'):
            if hasattr(source, name):
                setattr(policy, name, getattr(source, name))

    def step(epsilon):
        sync()
        return original_step(epsilon)

    sync()
    env.step = step
    return env


def build_environment(config, agent, device, ppo_shuffle_func, **kwargs):
    options = acceleration_settings()
    if not options['enabled']:
        return _speed_original_build_environment(config, agent, device, ppo_shuffle_func, **kwargs)
    import SatelliteNetworkSimulator_Computing as simulation
    mode = config['agent']['mode']
    algorithm = config['general']['algorithm_name']
    ppo = 'PPO' in mode
    plain = mode in ('New_DQN', 'Pure_DQN', 'Weak_DQN')
    known = algorithm.startswith('ISACR') or algorithm in ('DDQN', 'D3QN', 'NoSelfFeature')
    cpu = (options['cpu_policy'] and plain and known and agent is not None
           and torch.device(device).type == 'cuda'
           and config['general'].get('select_mode', 1) == 1
           and kwargs.get('experience_callback') is None)
    replay = (options['replay'] and not ppo and agent is not None
              and config['general']['phase'] == 'train'
              and isinstance(getattr(agent, 'replay_buffer', None), deque))
    if replay:
        agent.replay_buffer = IndexedReplay(agent.replay_buffer,
                                            maxlen=agent.replay_buffer.maxlen)
    compact_replay = replay and mode == 'GNN_DQN' and algorithm in ('GAT', 'GraphSAGE', 'JKNet')
    if compact_replay:
        from replay_storage import pack_experiences, collate
        agent.replay_graph_packer = pack_experiences
        agent.replay_graph_collate = collate
    # PPO retains its original routing kernel; all algorithms can defer pure
    # route rebuilding until a reader actually needs the derived tables.
    routes = options['routing'] and not ppo
    sampled_action = ppo and options['ppo_sampled_action']
    with _speed_lock:
        original = simulation.Satellite_with_Computing
        gnn = getattr(agent, 'online_gnn', None) if 'GNN' in mode else None
        satellite_base = getattr(gnn, 'satellite_class', original)
        attributes = {}
        # Snapshot messages change the original live-table delivery semantics.
        # Keep the shared sender and receiver untouched for every switch setting.
        if options['routing']:
            attributes['build_routing_table'] = _speed_defer_routes
            attributes['_speed_eager_routes'] = (_speed_routing_table if routes
                                                else satellite_base.build_routing_table)
            attributes['routing_tables'] = _speed_route_property('routing_tables')
            attributes['neighbor_hops'] = _speed_route_property('neighbor_hops')
        if sampled_action:
            attributes['get_next_hop'] = _speed_sampled_action_policy(satellite_base.get_next_hop)
        satellite_class = type('AcceleratedSatellite', (satellite_base,), attributes)
        with _speed_satellite_scope(simulation, gnn, satellite_class):
            build = _speed_build_cpu_environment if cpu else None
            if build is not None:
                env = build(_speed_original_build_environment, config, agent, device,
                            ppo_shuffle_func, **kwargs)
            else:
                env = _speed_original_build_environment(config, agent, device,
                                                        ppo_shuffle_func, **kwargs)
    # Training resets the environment after construction, and validation can
    # reset it again. Each reset constructs a new simulator and satellite class.
    if hasattr(env, 'reset'):
        original_reset = env.reset

        def reset(*args, **reset_kwargs):
            with _speed_satellite_scope(simulation, gnn, satellite_class):
                return original_reset(*args, **reset_kwargs)

        env.reset = reset
    env.acceleration_report = dict(runtime_acceleration_contract(), actual={
        'cpu_policy': cpu, 'routing': routes, 'replay': replay,
        'replay_storage': 'lossless_lz4_shared' if compact_replay else 'native',
        'ppo_sampled_action': sampled_action,
        'compact_messages': False,
        'lazy_routes': options['routing'],
        'policy_device': 'cpu' if cpu else str(device),
    })
    return env


def run_training(config):
    if not acceleration_settings()['enabled']:
        return _speed_original_run_training(config)
    config = copy.deepcopy(config)
    config['runtime_acceleration'] = runtime_acceleration_contract()
    try:
        return _speed_original_run_training(config)
    finally:
        # Pool workers run multiple seeds in sequence. SimPy generators and
        # environment callbacks form cycles; reclaim a finished job before
        # this worker allocates the next simulator and replay buffer.
        gc.collect()


def load_checkpoint_training_config(checkpoint_path):
    try:
        return _speed_original_load_checkpoint(checkpoint_path)
    except ValueError:
        # Existing models predate this appended layer. Accept only the exact
        # unchanged prefix hash; every other source, resource and weight hash
        # must still match. New models record the full current file hash.
        metadata_path = Path(str(checkpoint_path) + '.training.json')
        if not metadata_path.is_file():
            raise
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        config = metadata['config']
        recorded = config.get('training_input_sha256', {})
        current = training_input_hashes(config)
        changed = {key for key in recorded.keys() | current.keys()
                   if recorded.get(key) != current.get(key)}
        # Assemble the marker to avoid finding its string literal in this code.
        prefix = Path(__file__).read_bytes().split(
            b'# BEGIN OPTIONAL_RUNTIME_' + b'ACCELERATION_V1\n', 1)[0]
        if (changed != {'training_runner.py'}
                or recorded.get('training_runner.py') != hashlib.sha256(prefix).hexdigest()
                or metadata['checkpoint_sha256'] != checkpoint_content_hashes(checkpoint_path)):
            raise
        return config


def archive_test_outputs(paths):
    """Preserve prior raw results and contracts before an actual rerun."""
    paths = [Path(path) for path in paths if Path(path).is_file()]
    if not paths:
        return
    parent = paths[0].parent
    if any(path.parent != parent for path in paths):
        raise ValueError('Archived case files must share a directory')
    archive_root = parent / 'previous_attempts'
    archive_root.mkdir(exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix='attempt_', dir=archive_root))
    for path in paths:
        path.replace(target / path.name)
