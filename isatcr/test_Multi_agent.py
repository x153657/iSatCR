"""Evaluate GraphPR with GSJF-style one-hop computation offloading.

Run --dry-run to generate the 70-case plan without requiring checkpoints.
Use a new --output-root to preserve existing local-computation results.
Default checkpoint: training_process_data/v1/model/Multi_agent/GraphPR_seed{seed}.pth
The policy contract lives in train_Multi_agent.py; original routing-only DGL
checkpoints are deliberately not padded, truncated or silently partially loaded.
"""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
import textwrap
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / 'training_process_data/v1/test/test/Multi_agent'
LOAD_FAULT_PROBABILITY = 0.18
FAULT_LOAD_POINT = 160
ALGORITHM = 'GraphPR-GSJF'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_config(suite, seed, checkpoint, kind, value, raw_path):
    from train_Multi_agent import seed_config
    from training_runner import runtime_acceleration_contract

    config = seed_config(seed)
    config['runtime_acceleration'] = runtime_acceleration_contract()
    config['general'].update(suite.COMMON_TEST['general'])
    config['general'].update(algorithm_name='GraphPR', model_saving_enabled=False,
                             warmup_steps=0)
    config['test_contract'] = dict(
        algorithm=ALGORITHM, model_algorithm='GraphPR', compute_rule='gsjf_one_hop',
        method='gsjf_one_hop_offloading_then_graphpr_routing',
        neighbor_resource_source='received_periodic_state_messages',
        offload_hops=1,
        compute_target_selected_once=True, postcompute_route='original_GraphPR_policy',
        require_computed_before_downlink=True,
        offloading_source_sha256=digest(PROJECT_ROOT / 'test_GCSR.py'))
    # Evaluation random_seed remains common (2025); training_seed identifies
    # the independently trained checkpoint rather than changing test inputs.
    config['training_seed'] = seed
    config['agent'] = {'mode': 'Pure_DQN', 'shuffle': False,
                       'model_path': str(checkpoint), 'action_dim': 4,
                       'state_dim': 5 * config['graphpr']['node_feature_dim']
                                    + config['graphpr']['packet_feature_dim'] + 13}
    if kind == 'load':
        load = value
        probability = LOAD_FAULT_PROBABILITY
        case_id = value['id']
    else:
        load = suite.LOAD_SETTINGS_BY_POINT[FAULT_LOAD_POINT]
        probability = value
        case_id = suite.FAULT_CASE_BY_PROBABILITY[value]['id']
    # Both experiment axes share the same fixed trace library.
    workload_id = case_id if kind == 'load' else load['id']
    fault_id = suite.FAULT_CASE_BY_PROBABILITY[probability]['id']
    workload = suite.SCENARIO_MANIFEST['workloads'][workload_id]
    fault = suite.SCENARIO_MANIFEST['faults'][fault_id]
    config['environment'].update(suite.COMMON_TEST['environment'])
    config['environment']['require_computed_before_downlink'] = True
    config['environment']['uncomputed_reach_penalty'] = 0.0
    config['environment'].update(load)
    config['environment'].update(
        id=case_id, edge_p=probability, edge_q=1 - probability,
        save_training_data=str(raw_path),
        workload_trace_path=str(suite.resolve_trace(workload)),
        workload_trace_sha256=workload['sha256'],
        workload_trace_task_count=workload['task_count'],
        fault_trace_path=str(suite.resolve_trace(fault)),
        fault_trace_sha256=fault['sha256'],
        fault_trace_event_count=fault['event_count'],
    )
    return config


def compute_destination_forward(function):
    """Prepare the compute target before the RL destination/mask checks."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    inserted = 0
    for node in ast.walk(tree):
        for _, statements in ast.iter_fields(node):
            if not isinstance(statements, list):
                continue
            for index, statement in enumerate(list(statements)):
                if (isinstance(statement, ast.Expr) and ast.unparse(statement)
                        == "add_packet_delay(packet, 'computation', self.processing_time)"):
                    statements[index + 1:index + 1] = ast.parse(
                        "self.prepare_gosr_routing(packet)\n"
                        "if not is_computed:\n"
                        "    destination = packet._gsjf_target if packet._gsjf_target is not None else '_GSJF_NO_TARGET_'"
                    ).body
                    inserted += 1
    if inserted != 1:
        raise RuntimeError('GraphPR computation preparation hook changed')
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, __file__, 'exec'), function.__globals__, namespace)
    return namespace[function.__name__]


def choose_gsjf_target(node, packet, graph):
    """Compare local and one-hop completion time using GSJF's queue model."""
    import math
    import test_GCSR as gs

    candidates = []
    local = graph.nodes[node.name]
    estimate = gs.estimate_computation(local, packet.information[2], node.env.now)
    if (math.isfinite(estimate.total)
            and local.get('current_memory_occupy', 0.0) <= node._gosr_memory()):
        candidates.append((estimate.total, False, node.name))
    slots = getattr(node, 'action_neighbors', node.neighbors)
    if packet.hops < 2 * node.max_hop:
        for index in node._live_action_indices(packet.size):
            if index >= 4 or index >= len(slots):
                continue
            neighbor = slots[index]
            attrs = graph.nodes[neighbor]
            if (neighbor == packet.destination
                    or not node.neighbor_has_resource_state(neighbor)
                    or attrs.get('gosr_resource_missing', False)
                    or attrs.get('computing_ability', 0) <= 0
                    or not node._gosr_forward_capacity(neighbor)
                    or not node._gosr_can_hold(neighbor, packet.size, graph, attrs)
                    or graph[node.name][neighbor].get('missing', 0)):
                continue
            estimate = gs.estimate_neighbor(
                attrs, packet.information[2], packet.size, node.transmission_rate,
                send_waiting=node._gosr_edge_wait(neighbor),
                propagation=graph[node.name][neighbor].get('propagation_weight', 0.0),
                now=node.env.now)
            travel = estimate.send_waiting + estimate.transmission + estimate.propagation
            total = max(0.0, estimate.queueing - travel) + estimate.computation + travel
            if math.isfinite(total):
                candidates.append((total, True, neighbor))
    return min(candidates)[2] if candidates else None


@contextmanager
def greedy_offloading():
    import SatelliteNetworkSimulator_Computing as simulation
    import test_GCSR as gs
    base = simulation.Satellite_with_Computing
    propagator = simulation.Propagator_Computing

    class ResourceMessage(list):
        pass

    class Propagator(propagator):
        def send_state(self, node, neighbor, value, eliminated_self_feature=None):
            message = ResourceMessage(value)
            message.propagation_delay = self.propagation_delays.get((node, neighbor), 0.0)
            yield from super().send_state(node, neighbor, message, eliminated_self_feature)

    class Satellite(base):
        _gosr_graph = gs.GSJFSatellite._gosr_graph
        _gosr_edge_wait = gs.GSJFSatellite._gosr_edge_wait
        _gosr_memory = gs.GSJFSatellite._gosr_memory
        _gosr_can_hold = gs.GSJFSatellite._gosr_can_hold
        _gosr_forward_capacity = gs.GSJFSatellite._gosr_forward_capacity
        forward_packet = compute_destination_forward(base.forward_packet)

        def on_neighbor_state_received(self, neighbor, sent_at=None):
            super().on_neighbor_state_received(neighbor, sent_at=sent_at)
            message = self.neighbor_states[neighbor]
            if hasattr(message, 'propagation_delay'):
                if not hasattr(self, '_gosr_received_delays'):
                    self._gosr_received_delays = {}
                self._gosr_received_delays[neighbor] = message.propagation_delay

        def prepare_gosr_routing(self, packet):
            self._gsjf_packet = packet
            if packet.information[0]:
                return
            if not hasattr(packet, '_gsjf_target'):
                target = choose_gsjf_target(self, packet, self._gosr_graph())
                packet._gsjf_target = target
                packet._gsjf_origin = self.name
                packet._gsjf_origin_hops = packet.hops
                packet._gsjf_downlink = packet.destination
            assert packet.computing_node is None, 'RL offloading must not create Tradition PRE reservations'
            assert packet.destination == packet._gsjf_downlink

        def get_next_hop(self, current_state, destination, packet_size=None,
                         forced_action=None, prepared_policy_state=None,
                         prepared_action_mapping=None, *, hops, wait_count=0,
                         previous_hop=None, postcompute_previous_hop=None):
            packet = self._gsjf_packet
            if not packet.information[0]:
                target = packet._gsjf_target
                if target is None:
                    return None
                if target == self.name:
                    forced_action = 4
                else:
                    assert self.name == packet._gsjf_origin, 'Repeated offloading forbidden'
                    slots = self.slots()
                    if target not in self.neighbors or target not in slots:
                        return None
                    forced_action = slots.index(target)
            return super().get_next_hop(
                current_state, destination, packet_size, forced_action,
                prepared_policy_state, prepared_action_mapping, hops=hops,
                wait_count=wait_count, previous_hop=previous_hop,
                postcompute_previous_hop=postcompute_previous_hop)

        def push_computing(self, packet, demand):
            assert packet._gsjf_target == self.name
            assert packet.hops == packet._gsjf_origin_hops + (self.name != packet._gsjf_origin)
            return super().push_computing(packet, demand)

    with patch.object(simulation, 'Satellite_with_Computing', Satellite), \
         patch.object(simulation, 'Propagator_Computing', Propagator):
        yield


def run_case(config, checkpoint, raw_path):
    import random
    import numpy as np
    import torch
    from training_runner import build_environment
    from test_5ix_5seed_5sat import load_checkpoint_training_config
    from train_Multi_agent import GraphPRPolicy, graphpr_environment, acceleration_enabled

    seed = config['general']['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not acceleration_enabled() else 'cpu')
    training = load_checkpoint_training_config(checkpoint)
    if training.get('general', {}).get('algorithm_name') != 'GraphPR':
        raise ValueError('Checkpoint is not a GraphPR training result')
    if training['general']['random_seed'] != config['training_seed']:
        raise ValueError('Checkpoint training seed does not match the requested model')
    if training.get('method_sha256') != {'train_Multi_agent.py': digest(PROJECT_ROOT / 'train_Multi_agent.py')}:
        raise ValueError('GraphPR training/test method source mismatch')
    trained_method = dict(training.get('graphpr', {}))
    trained_method.pop('source_sha256', None)
    if trained_method != config['graphpr']:
        raise ValueError('GraphPR training/test feature or method contract mismatch')
    policy = GraphPRPolicy.load(checkpoint, device)

    class OffloadingInterface(torch.nn.Module):
        # Physical action 4 belongs to the heuristic; the learned model still
        # has exactly four outputs and never learns a computation action.
        def forward(self, state, valid_mask=None):
            return torch.cat((policy(state), state.new_full((state.shape[0], 1), -torch.inf)), dim=1)

        def aggregate(self, graph):
            return policy.aggregate(graph)

    agent = SimpleNamespace(online_net=OffloadingInterface().eval())
    with ExitStack() as stack:
        # Enable the existing evaluation-only physical fifth action without
        # changing the trained four-output model or its source contract.
        rule = config['test_contract']['compute_rule']
        config['test_contract']['compute_rule'] = 'local'
        try:
            stack.enter_context(graphpr_environment(config))
        finally:
            config['test_contract']['compute_rule'] = rule
        stack.enter_context(greedy_offloading())
        stack.enter_context(torch.no_grad())
        env = build_environment(config, agent, device, None)
        for _ in range(int(config['general']['duration']
                           / config['general']['time_stride'])):
            env.step(0.0)
    env.show_satellite_computing_time()
    env.show_test_trace_fingerprints()  # Also verifies complete fixed replay.
    if not raw_path.is_file():
        raise RuntimeError(f'No test statistics written: {raw_path}')


def checkpoint_for(model_root, template, seed):
    path = model_root / template.format(seed=seed)
    if path.exists():
        return path.resolve()
    best = sorted(model_root.glob(f'GraphPR_seed{seed}_train_reward_best*.pth'))
    if len(best) > 1:
        raise ValueError(f'Multiple reward-best checkpoints for seed {seed}: {best}')
    return best[0].resolve() if best else path.resolve()


def initialize_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_job(job):
    import test_4_5seed as suite
    os.chdir(PROJECT_ROOT)
    directory, config, checkpoint, contract, fingerprint = job
    raw_path = Path(config['environment']['save_training_data'])
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('fingerprint') != fingerprint:
            raise RuntimeError(f'Existing result contract changed; use a new output root: {directory}')
        if not raw_path.is_file() or digest(raw_path) != summary.get('raw_sha256'):
            raise RuntimeError(f'Existing raw result is missing or changed: {raw_path}')
        return summary['result']
    if raw_path.exists():
        raise RuntimeError(f'Partial output exists; use a new output root: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    suite.write_json(directory / 'config.json', config)
    suite.write_json(directory / 'contract.json', contract)
    suite.write_json(directory / 'STATUS.json', {'state': 'running'})
    try:
        run_case(config, checkpoint, raw_path)
        load = config['environment']
        result = suite.parse_test_result(raw_path, ALGORITHM, load,
                                         config['general']['duration'])
        result.update(training_seed=config['training_seed'],
                      scenario_seed=config['general']['scenario_seed'],
                      test_type=directory.parent.name, case_id=directory.name,
                      fault_percent=round(load['edge_p'] * 100, 8),
                      model_path=str(checkpoint))
        suite.write_json(summary_path, {'fingerprint': fingerprint,
                         'raw_sha256': digest(raw_path), 'result': result})
        suite.write_json(directory / 'STATUS.json', {'state': 'completed'})
        return result
    except Exception as error:
        suite.write_json(directory / 'STATUS.json', {'state': 'failed', 'error': str(error)})
        raise


def reuse_job(job, source_job):
    """Materialize a second experiment-axis row without repeating the simulation."""
    import test_4_5seed as suite
    directory, config, checkpoint, contract, fingerprint = job
    source_directory, source_config, _, _, source_fingerprint = source_job
    source_raw = Path(source_config['environment']['save_training_data'])
    source = json.loads((source_directory / 'summary.json').read_text(encoding='utf-8'))
    if source['fingerprint'] != source_fingerprint or digest(source_raw) != source['raw_sha256']:
        raise RuntimeError(f'Invalid intersection source: {source_directory}')
    summary_path = directory / 'summary.json'
    raw_path = Path(config['environment']['save_training_data'])
    if summary_path.exists():
        stored = json.loads(summary_path.read_text(encoding='utf-8'))
        if (stored.get('fingerprint') != fingerprint or not raw_path.is_file()
                or digest(raw_path) != stored.get('raw_sha256')):
            raise RuntimeError(f'Existing intersection contract changed: {directory}')
        return stored['result']
    if raw_path.exists():
        raise RuntimeError(f'Partial intersection output exists: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_raw, raw_path)
    result = dict(source['result'], test_type=directory.parent.name,
                  case_id=directory.name, raw_result_file=str(raw_path))
    suite.write_json(directory / 'config.json', config)
    suite.write_json(directory / 'contract.json', contract)
    suite.write_json(summary_path, dict(fingerprint=fingerprint,
                     raw_sha256=digest(raw_path), result=result,
                     reused_from=str(source_directory)))
    suite.write_json(directory / 'STATUS.json', dict(state='completed',
                     reused_from=str(source_directory)))
    return result


def physical_job_groups(jobs):
    groups = {}
    for job in jobs:
        config = job[1]
        environment = dict(config['environment'])
        # These describe output/axis identity rather than the physical scenario.
        environment.pop('id', None)
        environment.pop('save_training_data', None)
        key = (config['training_seed'], json.dumps(environment, sort_keys=True))
        groups.setdefault(key, []).append(job)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--model-root', type=Path,
                        default=PROJECT_ROOT / 'training_process_data/v1/model/Multi_agent')
    parser.add_argument('--model-template', default='GraphPR_seed{seed}.pth')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config-root', type=Path,
                        default=PROJECT_ROOT / 'training_process_data/v1/test/test/config')
    parser.add_argument('--test-types', nargs='+', choices=['load', 'fault'],
                        default=['load', 'fault'])
    parser.add_argument('--load-points', type=int, nargs='+')
    parser.add_argument('--fault-percentages', type=int, nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error('--workers must be between 1 and 10')
    if len(set(args.seeds)) != len(args.seeds) or any(s not in range(5) for s in args.seeds):
        parser.error('Seeds must be unique members of 0..4')

    os.chdir(PROJECT_ROOT)
    os.environ['SATELLITE_TEST_CONFIG_ROOT'] = str(args.config_root.resolve())
    startup_hash_seed = os.environ.get('PYTHONHASHSEED')
    import test_4_5seed as suite
    suite.validate_fixed_configs()
    if ({int(x['target_load_tasks_per_second']) for x in suite.LOAD_SETTINGS}
            != set(range(100, 161, 10))
            or FAULT_LOAD_POINT not in suite.LOAD_SETTINGS_BY_POINT
            or LOAD_FAULT_PROBABILITY not in suite.FAULT_CASE_BY_PROBABILITY
            or {round(p * 100) for p in suite.FAULT_PROBABILITIES}
            != set(range(0, 19, 3))):
        raise ValueError('Fixed configs must match load 100..160 at 18%, fault 0..18% at load 160')
    # Deterministic hash iteration must be set before interpreter startup.
    if not args.dry_run and startup_hash_seed != suite.PYTHON_HASH_SEED:
        raise RuntimeError('Set PYTHONHASHSEED to the common.json value before starting')
    output = args.output_root.resolve()
    source_hashes = {name: digest(PROJECT_ROOT / name) for name in (
        'train_Multi_agent.py', 'test_Multi_agent.py',
        'SatelliteNetworkSimulator_Computing.py', 'RL_environment_for_computing.py',
        'SatelliteNetworkSimulator_Beta.py', 'training_runner.py', 'test_4_5seed.py',
        'test_TBO-SP.py', 'test_GCSR.py',
        'test_5ix_5seed_5sat.py',
    )}
    loads = sorted(suite.LOAD_SETTINGS, key=lambda x: x['target_load_tasks_per_second'],
                   reverse=True)
    faults = sorted(suite.FAULT_PROBABILITIES, reverse=True)
    available_loads = {int(x['target_load_tasks_per_second']) for x in loads}
    available_faults = {int(round(p * 100)) for p in faults}
    if args.load_points and not set(args.load_points) <= available_loads:
        parser.error(f'Load points must be in {sorted(available_loads)}')
    if args.fault_percentages and not set(args.fault_percentages) <= available_faults:
        parser.error(f'Fault percentages must be in {sorted(available_faults)}')
    cases = []
    if 'load' in args.test_types:
        cases += [('load', x) for x in loads if args.load_points is None
                  or int(x['target_load_tasks_per_second']) in args.load_points]
    if 'fault' in args.test_types:
        cases += [('fault', p) for p in faults if args.fault_percentages is None
                  or int(round(p * 100)) in args.fault_percentages]
    jobs = []
    for seed in args.seeds:
        checkpoint = checkpoint_for(args.model_root.resolve(), args.model_template, seed)
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(f'Retrained GraphPR checkpoint required: {checkpoint}')
        for kind, value in cases:
            case_id = value['id'] if kind == 'load' else suite.FAULT_CASE_BY_PROBABILITY[value]['id']
            directory = output / f'seed{seed}' / kind / case_id
            config = build_config(suite, seed, checkpoint, kind, value,
                                  directory / 'raw.txt')
            contract = {'config': config, 'source_hashes': source_hashes,
                        'configuration_hashes': suite.CONFIG_HASHES,
                        'checkpoint_sha256': digest(checkpoint) if checkpoint.is_file() else None}
            fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            saved_summary = directory / 'summary.json'
            if saved_summary.exists() and json.loads(saved_summary.read_text(
                    encoding='utf-8')).get('fingerprint') != fingerprint:
                raise RuntimeError(f'Existing result contract changed; use a new --output-root: {directory}')
            jobs.append((directory, config, checkpoint, contract, fingerprint))
    output.mkdir(parents=True, exist_ok=True)
    groups = physical_job_groups(jobs)
    plan = {'algorithm': ALGORITHM, 'model_algorithm': 'GraphPR',
            'load_experiment_fault_probability': LOAD_FAULT_PROBABILITY,
            'fault_experiment_load': FAULT_LOAD_POINT,
            'seeds': args.seeds, 'case_count': len(jobs),
            'physical_simulation_count': len(groups),
            'dry_run': args.dry_run, 'execution': 'parallel', 'workers': args.workers,
            'cases': [dict(config=config, contract=contract, fingerprint=fingerprint)
                      for _, config, _, contract, fingerprint in jobs]}
    suite.write_json(output / ('TEST_PLAN.json' if args.dry_run else 'RUN_MANIFEST.json'), plan)
    if args.dry_run:
        print(f'Validated {len(jobs)} Walker cases; plan: {output / "TEST_PLAN.json"}')
        return

    lock = output / 'TEST_RUNNING.lock'
    with lock.open('x', encoding='utf-8') as file:
        file.write(str(os.getpid()))
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as executor:
            futures = {executor.submit(run_job, group[0]): group for group in groups}
            for future in as_completed(futures):
                rows.append(future.result())
                group = futures[future]
                rows.extend(reuse_job(job, group[0]) for job in group[1:])
                rows.sort(key=lambda r: (r['training_seed'], r['test_type'], r['case_id']))
                print(f'[{len(rows)}/{len(jobs)}] completed', flush=True)
                suite.write_rows(output / 'all_seed_results.csv', rows)
                for kind in ('load', 'fault'):
                    selected = [r for r in rows if r['test_type'] == kind]
                    if selected:
                        suite.write_rows(output / f'five_seed_{kind}_mean.csv', suite.mean_rows(
                            selected, ['test_type', 'case_id', 'target_load_tasks_per_second', 'fault_percent']))
                suite.write_json(output / 'TEST_STATUS.json', {
                    'state': 'completed' if len(rows) == len(jobs) else 'running',
                    'completed_cases': len(rows), 'expected_cases': len(jobs),
                    'workers': args.workers,
                })
    except Exception as error:
        suite.write_json(output / 'TEST_STATUS.json', {'state': 'failed',
                         'completed_cases': len(rows), 'expected_cases': len(jobs),
                         'error': str(error)})
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
