"""Fixed test configs, destination selection and shared test acceleration."""

import argparse
import ast
import gc
import inspect
import textwrap
import weakref
from collections.abc import Mapping
from functools import wraps
import numpy as np
from contextlib import contextmanager
import os
import sys
import datetime as dt
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import geopandas as gpd
import networkx as nx
from skyfield.api import load
from shapely.geometry import MultiPolygon, Point, Polygon

from Make_Satellite_Graph import SatelliteGraph, SatelliteTracker
from Read_Ground_Imformation import extract_landmarks, get_connections_h3


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = (
    PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test' / 'config'
)

# The trace fixes the session target.  Once a packet reaches that target
# satellite, it can downlink without mandatory computation;
# choosing another destination is not part of a test.
DESTINATION_POLICY = 'fixed_session_destination_direct_downlink_at_target'

CONSTELLATION_TLE_FILES = {
    'OneWeb': './Satellite_Data/OneWeb_tles.txt',
    'Starlink': './Satellite_Data/Starlink_tles.txt',
    'Telesat': './Satellite_Data/Telesat_tles.txt',
}

GENERAL = {
    # Fixed evaluation configuration: epsilon=0 forces greedy inference.
    # These values intentionally differ from training defaults.
    'phase': 'test',
    'random_seed': 2025,
    'epsilon': 0.0,
    'epsilon_decay': 0.999,
    'min_epsilon': 0.02,
    'begin_time': '2023-10-01 00:00:00',
    'time_stride': 1,
    'rounds': 1,
    'skip_time': [60, 0],
    'duration': 300,
    'reward_factors': [1, 0.05, 1, 0.1, 0.25],
    'print_cycle': 1,
    'select_mode': 1,
    'show_test_summary': True,
}

ENVIRONMENT = {
    'mission_possibility': [0.5, 0.5],
    'computing_demand_factor': [150, 250],
    'computing_demand_factor_2': [300, 500],
    'size_after_computing_factor': [0.0909, 0.1111],
    'size_after_computing_1': 40000.0,
    'packet_frequency': 1,
    'mean_interval_time': 30,
    'memory': 8e9,
    # Per-satellite values are required from the TLE companion CSV.
    'computing_ability': None,
    'transmission_rate': 1.2e9,
    'downlink_rate': 3e9,
    'downstream_delays': 0.0016667,
    'packet_size_range': [200e6, 600e6],
    'state_update_period': 0.1,
    'heartbeat_timeout': 0.15,
    'del_cycle': 30,
    'update_cycle': 10,
    'markov_mode': True,
    'node_p': 0.0,
    'node_q': 0.0,
    'fault_compute_double_mean': True,
    'fault_fill_multiplier': 2.0,
    'require_computed_before_downlink': False,
    'visualize': False,
    'print_info': False,
    'save_log': False,
    'show_detail': False,
    'random_edges_del': 0,
    'random_nodes_del': 0,
    'elevation_angle': 45,
    'pole': False,
    'tle_filepath': './Satellite_Data/wallker_tles.txt',
    'SOD_file_path': './Ground_Data/11_ground_stations.txt',
    'world_path': './ne_data/ne_50m_admin_0_countries_lakes.dbf',
}

LOAD_CASES = [
    {'id': 'load_100', 'target_load_tasks_per_second': 100, 'poisson_rate_g': 34.98, 'poisson_rate_s': 69.97},
    {'id': 'load_110', 'target_load_tasks_per_second': 110, 'poisson_rate_g': 30.00, 'poisson_rate_s': 60.00},
    {'id': 'load_120', 'target_load_tasks_per_second': 120, 'poisson_rate_g': 25.86, 'poisson_rate_s': 51.72},
    {'id': 'load_130', 'target_load_tasks_per_second': 130, 'poisson_rate_g': 22.37, 'poisson_rate_s': 44.74},
    {'id': 'load_140', 'target_load_tasks_per_second': 140, 'poisson_rate_g': 19.39, 'poisson_rate_s': 38.78},
    {'id': 'load_150', 'target_load_tasks_per_second': 150, 'poisson_rate_g': 16.82, 'poisson_rate_s': 33.63},
    {'id': 'load_160', 'target_load_tasks_per_second': 160, 'poisson_rate_g': 14.57, 'poisson_rate_s': 29.15},
]

FAULT_CASES = [
    {'id': f'fault_p{str(p).replace(".", "p")}', 'fault_percent': int(p * 100), 'edge_p': p, 'edge_q': 1.0 - p}
    for p in (0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18)
]
FAULT_CASES[0]['id'] = 'fault_p0'
# The five-coefficient multi-constellation runner uses load 160/fault 18%.
# Keep it separate from the regular load matrix.
MULTI_CONSTELLATION_LOAD_CASE = {
    'id': 'load_160',
    'target_load_tasks_per_second': 160,
    'poisson_rate_g': 14.57,
    'poisson_rate_s': 29.15,
}

COEFFICIENTS = (1.0, 1.125, 1.25, 1.375, 1.5)


def canonical_edge(left, right):
    return tuple(sorted((str(left), str(right))))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')


def write_bytes(path, content, force):
    if path.exists() and not force:
        raise FileExistsError(f'{path} already exists; use --force to replace it')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def write_json(path, value, force):
    content = json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8') + b'\n'
    return write_bytes(path, content, force)


def skyfield_time(timescale, begin, seconds):
    current = begin + dt.timedelta(seconds=float(seconds))
    return timescale.utc(
        current.year, current.month, current.day,
        current.hour, current.minute, current.second + current.microsecond / 1e6,
    )


def satellites_over_region(world, positions):
    satellites = set()
    geometries = world.geometry
    spatial_index = world.sindex
    for satellite, coordinates in positions.items():
        point = Point(coordinates['longitude'], coordinates['latitude'])
        for index in spatial_index.query(point):
            geometry = geometries.iloc[index]
            if isinstance(geometry, (Polygon, MultiPolygon)) and geometry.contains(point):
                satellites.add(satellite)
                break
    return satellites


def build_nominal_context(environment=ENVIRONMENT):
    tle_path = PROJECT_ROOT / environment['tle_filepath']
    ground_path = PROJECT_ROOT / environment['SOD_file_path']
    tracker = SatelliteTracker(str(tle_path))
    graph_builder = SatelliteGraph()
    landmarks = extract_landmarks(str(ground_path))
    world = gpd.read_file(PROJECT_ROOT / environment['world_path'])
    timescale = load.timescale()
    begin = dt.datetime.strptime(GENERAL['begin_time'], '%Y-%m-%d %H:%M:%S')
    snapshots = {}
    graph = None
    for second in range(0, GENERAL['duration'] + 1, ENVIRONMENT['update_cycle']):
        time = skyfield_time(timescale, begin, second)
        graph = graph_builder.build_graph_with_fixed_edges(tracker, time, pole=False)
        positions = tracker.generate_satellite_LLA_dict(time)
        connections = get_connections_h3(
            landmarks, positions, environment['elevation_angle']
        )
        direct = sorted({sat for values in connections.values() for sat in values})
        snapshots[second] = {
            'graph': graph,
            'direct': direct,
            'ground_cover': satellites_over_region(world, positions),
        }
    return tracker, graph_builder, timescale, begin, snapshots


def snapshot_at(snapshots, second):
    cycle = ENVIRONMENT['update_cycle']
    tick = min(int(math.floor(second / cycle)) * cycle, GENERAL['duration'])
    return snapshots[tick]


def select_session_destination(seed, satellite, session_index, snapshot):
    graph = snapshot['graph']
    if satellite not in graph:
        raise RuntimeError(f'Unknown workload source satellite: {satellite}')
    hop_counts = nx.single_source_shortest_path_length(graph, satellite)
    reachable = [
        destination for destination in snapshot['direct']
        if destination in hop_counts
    ]
    if not reachable:
        return None
    minimum_hops = min(hop_counts[destination] for destination in reachable)
    candidates = sorted(
        destination for destination in reachable
        if hop_counts[destination] == minimum_hops
    )
    destination_rng = random.Random(
        f'destination:{seed}:{satellite}:{session_index}'
    )
    return destination_rng.choice(candidates)


def generate_workload(seed, setting, snapshots):
    sessions = []
    duration = GENERAL['duration']
    frequency = ENVIRONMENT['packet_frequency']
    satellites = sorted(snapshots[0]['graph'].nodes())
    for satellite in satellites:
        rng = random.Random(f'traffic:{seed}:{satellite}')
        now = 0.0
        session_index = 0
        while now < duration:
            snapshot = snapshot_at(snapshots, now)
            rate = (
                setting['poisson_rate_g']
                if satellite in snapshot['ground_cover']
                else setting['poisson_rate_s']
            )
            now += min(rng.expovariate(1.0 / rate), rate * 3)
            if now > duration:
                break
            task_type = rng.choices([0, 1], weights=ENVIRONMENT['mission_possibility'], k=1)[0]
            snapshot = snapshot_at(snapshots, now)
            if satellite in snapshot['direct']:
                continue
            next_session_index = session_index + 1
            destination = select_session_destination(
                seed, satellite, next_session_index, snapshot
            )
            if destination is None:
                continue
            session_duration = min(
                rng.expovariate(1.0 / ENVIRONMENT['mean_interval_time']),
                ENVIRONMENT['mean_interval_time'] * 3,
            )
            session_start = now
            sampled_session_end = session_start + session_duration
            packets = []
            while now < sampled_session_end:
                packet_time = now + 1.0 / frequency
                if packet_time > duration:
                    now = float(duration)
                    break
                now = packet_time
                size = rng.randint(
                    int(ENVIRONMENT['packet_size_range'][0]),
                    int(ENVIRONMENT['packet_size_range'][1]),
                )
                if task_type == 0:
                    size_after = int(rng.uniform(*ENVIRONMENT['size_after_computing_factor']) * size)
                    demand = int(rng.uniform(*ENVIRONMENT['computing_demand_factor']) * size)
                else:
                    size_after = int(ENVIRONMENT['size_after_computing_1'])
                    demand = int(rng.uniform(*ENVIRONMENT['computing_demand_factor_2']) * size)
                packets.append({
                    'packet_index': len(packets) + 1,
                    'packet_size': size,
                    'computing_demand': demand,
                    'size_after_computing': size_after,
                })
            session_index = next_session_index
            sessions.append({
                'satellite': satellite,
                'destination': destination,
                'session_index': session_index,
                'session_start_time': round(session_start, 9),
                'session_end_time': round(min(now, duration), 9),
                'task_type': task_type,
                'packets': packets,
            })
    sessions.sort(key=lambda item: (item['satellite'], item['session_index']))
    return sessions


def graph_edges(graph):
    return {canonical_edge(left, right) for left, right in graph.edges()}


def generate_faults(seed, fault_case, tracker, graph_builder, timescale, begin):
    rng = random.Random(f'fault:{seed}')
    failed = set()
    events = []
    cycle = ENVIRONMENT['del_cycle']
    for second in range(0, GENERAL['duration'], cycle):
        graph = graph_builder.build_graph_with_fixed_edges(
            tracker, skyfield_time(timescale, begin, second), pole=False
        )
        available = graph_edges(graph)
        previous = set(failed)
        alive = sorted(available - previous)
        new_failures = {
            edge for edge in alive if rng.random() < fault_case['edge_p']
        }
        recoveries = {
            edge for edge in sorted(previous)
            if rng.random() < fault_case['edge_q']
        }
        failed = (previous - recoveries) | new_failures
        for edge in sorted(previous - failed):
            events.append({'time': second, 'action': 'recover', 'edge': list(edge)})
        for edge in sorted(failed - previous):
            events.append({'time': second, 'action': 'fail', 'edge': list(edge)})
    return events


def encode_jsonl(records):
    return b''.join(json_bytes(record) for record in records)


def generate(seed, context, force, config_root=CONFIG_ROOT,
             load_cases=LOAD_CASES, fault_cases=FAULT_CASES,
             constellation=None, environment=ENVIRONMENT):
    tracker, graph_builder, timescale, begin, snapshots = context
    scenario_root = config_root / f'scenario_seed_{seed}'
    tle_path = (PROJECT_ROOT / environment['tle_filepath']).resolve()
    computing_ability_path = tle_path.with_name(
        f'{tle_path.stem}_computing_ability.csv'
    )
    if not computing_ability_path.is_file():
        raise FileNotFoundError(computing_ability_path)
    manifest = {
        'schema_version': 2,
        'constellation': constellation,
        'tle_filepath': environment['tle_filepath'],
        'tle_sha256': hashlib.sha256(tle_path.read_bytes()).hexdigest(),
        'computing_ability_filepath': str(
            Path(environment['tle_filepath']).with_name(
                f"{Path(environment['tle_filepath']).stem}_computing_ability.csv"
            )
        ).replace('\\', '/'),
        'computing_ability_sha256': hashlib.sha256(
            computing_ability_path.read_bytes()
        ).hexdigest(),
        'satellite_count': len(tracker.satellites),
        'scenario_seed': seed,
        'duration': GENERAL['duration'],
        'event_order_at_same_time': ['fault_update', 'task_arrival'],
        'workloads': {},
        'faults': {},
    }
    for setting in load_cases:
        sessions = generate_workload(seed, setting, snapshots)
        path = scenario_root / 'workload' / f"{setting['id']}.jsonl"
        digest = write_bytes(path, encode_jsonl(sessions), force)
        type_counts = Counter(session['task_type'] for session in sessions)
        task_count = sum(len(session['packets']) for session in sessions)
        manifest['workloads'][setting['id']] = {
            'path': str(path.relative_to(config_root)).replace('\\', '/'),
            'sha256': digest,
            'satellite_count': len({session['satellite'] for session in sessions}),
            'session_count': len(sessions),
            'task_count': task_count,
            'type_0_session_count': type_counts[0],
            'type_1_session_count': type_counts[1],
        }
    for fault_case in fault_cases:
        events = generate_faults(
            seed, fault_case, tracker, graph_builder, timescale, begin
        )
        path = scenario_root / 'fault' / f"{fault_case['id']}.jsonl"
        digest = write_bytes(path, encode_jsonl(events), force)
        actions = Counter(event['action'] for event in events)
        manifest['faults'][fault_case['id']] = {
            'path': str(path.relative_to(config_root)).replace('\\', '/'),
            'sha256': digest,
            'event_count': len(events),
            'fail_count': actions['fail'],
            'recover_count': actions['recover'],
        }
    write_json(scenario_root / 'manifest.json', manifest, force)
    return manifest


def write_top_level_configs(seeds, force, config_root=CONFIG_ROOT,
                            environment=ENVIRONMENT, load_cases=LOAD_CASES,
                            fault_cases=FAULT_CASES,
                            fault_workload=None):
    if fault_workload is None:
        fault_workload = next(case for case in load_cases if case['id'] == 'load_160')
    common = {
        'schema_version': 2,
        'python_hash_seed': str(GENERAL['random_seed']),
        'scenario_seeds': seeds,
        'general': {
            **GENERAL,
            'random_seed': GENERAL['random_seed'],
            'scenario_seed': seeds[0],
        },
        'environment': environment,
        'stream_contract': {
            'mode': 'fixed_trace_files',
            'workload_schema': 'satellite_session_packets_v2_destination',
            'packet_interval_seconds': 1.0 / ENVIRONMENT['packet_frequency'],
            'destination_is_algorithm_decision': False,
            'destination_policy': DESTINATION_POLICY,
            'event_order_at_same_time': ['fault_update', 'task_arrival'],
            'require_workload_trace_match': True,
            'require_fault_trace_match': True,
        },
    }
    load_config = {
        'schema_version': 2,
        'test_type': 'load',
        'edge_p': 0.18,
        'edge_q': 0.82,
        'cases': load_cases,
    }
    fault_config = {
        'schema_version': 2,
        'test_type': 'fault',
        'workload': fault_workload,
        'cases': fault_cases,
    }
    write_json(config_root / 'common.json', common, force)
    write_json(config_root / 'load_experiments.json', load_config, force)
    write_json(config_root / 'fault_experiments.json', fault_config, force)


def write_coefficient_config(config_root, force):
    write_json(
        config_root / 'coefficient_experiments.json',
        {'schema_version': 1, 'coefficients': list(COEFFICIENTS)},
        force,
    )


def write_scenario_index(config_root, seeds, force):
    index = {
        'schema_version': 1,
        'scenario_seeds': seeds,
        'manifests': [f'scenario_seed_{seed}/manifest.json' for seed in seeds],
    }
    write_json(config_root / 'scenario_index.json', index, force)


def generate_configuration_set(
    config_root,
    environment,
    scenario_seeds,
    load_cases,
    fault_cases,
    force,
    constellation=None,
):
    context = build_nominal_context(environment)
    fault_workload = (
        next(case for case in load_cases if case['id'] == 'load_160')
    )
    write_top_level_configs(
        scenario_seeds,
        force,
        config_root,
        environment,
        load_cases,
        fault_cases,
        fault_workload,
    )
    trace_load_cases = list(load_cases)
    if not any(int(case['target_load_tasks_per_second']) == int(fault_workload['target_load_tasks_per_second']) for case in trace_load_cases):
        trace_load_cases.append(dict(
            fault_workload,
            id=f"load_{int(fault_workload['target_load_tasks_per_second'])}",
        ))
    manifests = [
        generate(
            seed,
            context,
            force,
            config_root,
            trace_load_cases,
            fault_cases,
            constellation,
            environment,
        )
        for seed in scenario_seeds
    ]
    write_scenario_index(config_root, scenario_seeds, force)
    for manifest in manifests:
        print(
            f"constellation={constellation or 'Walker'} "
            f"scenario_seed={manifest['scenario_seed']} "
            f"workloads={len(manifest['workloads'])} "
            f"faults={len(manifest['faults'])}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario-seeds', type=int, nargs='+', default=[2025])
    parser.add_argument(
        '--force',
        action='store_true',
        default=True,
        help='Replace existing configuration and trace files (default).',
    )
    parser.add_argument(
        '--no-force',
        dest='force',
        action='store_false',
        help='Keep existing files and fail if a target already exists.',
    )
    args = parser.parse_args()
    if len(set(args.scenario_seeds)) != len(args.scenario_seeds):
        parser.error('scenario seeds must be unique')

    # Task 1: the Walker constellation gets all load and fault experiment
    # points used by the four-algorithm fixed-trace tests.
    generate_configuration_set(
        CONFIG_ROOT,
        dict(ENVIRONMENT),
        args.scenario_seeds,
        LOAD_CASES,
        FAULT_CASES,
        args.force,
    )
    write_coefficient_config(CONFIG_ROOT, args.force)

    # Task 2: each constellation gets only the shared load-160/fault-18% case
    # used by the multi-constellation experiment.
    load_160 = (MULTI_CONSTELLATION_LOAD_CASE,)
    fault_18 = tuple(
        case for case in FAULT_CASES
        if math.isclose(case['edge_p'], 0.18)
    )
    if len(load_160) != 1 or len(fault_18) != 1:
        raise RuntimeError('Expected load 160 and fault 18% configuration cases')
    for constellation, tle_filepath in {'Walker': ENVIRONMENT['tle_filepath'], **CONSTELLATION_TLE_FILES}.items():
        environment = dict(ENVIRONMENT)
        environment['tle_filepath'] = tle_filepath
        generate_configuration_set(
            CONFIG_ROOT / 'sat' / constellation,
            environment,
            args.scenario_seeds,
            load_160,
            fault_18,
            args.force,
            constellation,
        )


@contextmanager
def destination_selection(ablation=False):
    """Install joint destination/action selection only inside a plus test worker."""
    import numpy as np
    import torch
    import SatelliteNetworkSimulator_Computing as simulation

    original = simulation.Satellite_with_Computing
    class DestinationSelectingSatellite(simulation.Satellite_with_Computing):
        def trans_parameters(self, q_net, reward_function, direct_satellites,
                             gnn=None, ppo_shuffle_func=None):
            super().trans_parameters(q_net, reward_function, direct_satellites,
                                     gnn, ppo_shuffle_func)
            self._destination_decision = None
            self._destination_shuffle = ppo_shuffle_func
            if ppo_shuffle_func is not None and 'GNN' not in self.mode:
                self.ppo_shuffle_func = self._shuffle_destination_state

        def _shuffle_destination_state(self, state):
            decision = self._destination_decision
            if decision is not None and np.array_equal(state, decision['state']):
                return decision['policy_state'], decision['mapping']
            return self._destination_shuffle(state)

        def find_min_hops_destinations(self, n):
            # Candidate identities come from ground coverage, not remote resources.
            # A packet already at its destination retains the ordinary downlink path.
            return sorted(destination for destination in self.direct_satellites
                          if destination != self.name)

        @torch.no_grad()
        def find_highest_score(self, destinations, mission_state, hops, is_computed):
            self._destination_decision = None
            mission_state = [mission_state[0], mission_state[1] / self.max_size,
                             simulation._normalized_computing_demand(mission_state[2]),
                             mission_state[3] / self.max_size]
            rows = []
            mapping = np.arange(4)
            for destination in destinations:
                if destination not in self.routing_tables:
                    continue
                mask = self.valid_action_mask(destination, mission_state[1] * self.max_size,
                                              is_computed, hops)
                if not mask.any():
                    continue
                state = self.get_current_state(destination, hops, is_computed, mission_state)
                if 'GNN' in self.mode:
                    # Ablations already store the communicated embeddings in physical
                    # slot order. Keep that order for both scoring and forced action.
                    policy_state = state
                    policy_mask = mask
                    tensor = torch.cat([
                        state[0].decision_embeddings.to(self.device),
                        torch.as_tensor(state[1], dtype=torch.float,
                                        device=self.device).unsqueeze(0),
                    ], dim=-1).squeeze(0)
                else:
                    if not rows and self._destination_shuffle is not None:
                        _, mapping = self._destination_shuffle(state)
                        mapping = np.asarray(mapping, dtype=int)
                    slot_width = simulation.NEIGHBOR_SLOT_DIM if self.mode == 'New_DQN' else 8
                    neighbor_width = 4 * slot_width
                    chunks = state[:neighbor_width].reshape(4, -1)
                    policy_state = np.concatenate([chunks[mapping].reshape(-1),
                                                   state[neighbor_width:]])
                    policy_mask = self.replay_action_mask(mask, mapping)
                    tensor = torch.as_tensor(policy_state, dtype=torch.float, device=self.device)
                rows.append((destination, state, policy_state, policy_mask, tensor))
            if not rows:
                return destinations[:1]
            states = torch.stack([row[4] for row in rows])
            masks = torch.tensor(np.asarray([row[3] for row in rows]),
                                 dtype=torch.bool, device=self.device)
            scores = self.q_net(states, masks).masked_fill(~masks, float('-inf'))
            flat_index = int(scores.argmax().item())
            row_index, policy_action = divmod(flat_index, scores.shape[1])
            destination, state, policy_state, _, _ = rows[row_index]
            action = int(mapping[policy_action]) if policy_action < 4 else policy_action
            self._destination_decision = {
                'destination': destination, 'state': state, 'policy_state': policy_state,
                'mapping': mapping, 'action': action,
            }
            return [destination]

        def get_next_hop(self, current_state, destination, packet_size=None,
                         forced_action=None, prepared_policy_state=None,
                         prepared_action_mapping=None, *, hops, wait_count=0,
                         previous_hop=None, postcompute_previous_hop=None):
            decision = self._destination_decision
            if decision is not None and decision['destination'] == destination:
                if forced_action is None:
                    mask = self.valid_action_mask(
                        destination, packet_size,
                        current_state[1][-1] if 'GNN' in self.mode else current_state[-1], hops,
                        wait_count, previous_hop, postcompute_previous_hop,
                    )
                    if mask[decision['action']]:
                        forced_action = decision['action']
                prepared_policy_state = decision['policy_state']
                prepared_action_mapping = decision['mapping']
            try:
                return super().get_next_hop(
                    current_state, destination, packet_size, forced_action,
                    prepared_policy_state, prepared_action_mapping, hops=hops,
                    wait_count=wait_count,
                    previous_hop=previous_hop,
                    postcompute_previous_hop=postcompute_previous_hop,
                )
            finally:
                self._destination_decision = None

    encoder_classes = []
    if ablation:
        from ablation.train_graphsage import GraphSAGEEncoder
        from ablation.message_aggregation import MessageReplay, AlignedDirectConcatEncoder
        for encoder in (GraphSAGEEncoder, MessageReplay, AlignedDirectConcatEncoder):
            base = encoder.satellite_class
            encoder_classes.append((encoder, base))
            encoder.satellite_class = type(
                'DestinationSelecting' + base.__name__,
                (DestinationSelectingSatellite, base), {},
            )
    simulation.Satellite_with_Computing = DestinationSelectingSatellite
    try:
        yield
    finally:
        simulation.Satellite_with_Computing = original
        for encoder, base in encoder_classes:
            encoder.satellite_class = base



# 四类测试共用的默认开关；命令行 --acceleration on/off 可单次覆盖。
TEST_ACCELERATION_ENABLED = True
_installed = False
_recipe = None


def acceleration_enabled():
    return os.environ.get('SATELLITE_TEST_ACCELERATION',
                          '1' if TEST_ACCELERATION_ENABLED else '0') == '1'


def add_acceleration_argument(parser):
    parser.add_argument('--acceleration', choices=('on', 'off'),
                        default='on' if acceleration_enabled() else 'off',
                        help='Enable/disable test acceleration; inherited by workers.')


def configure_acceleration(enabled):
    if _installed and not enabled:
        raise RuntimeError('Use a fresh process to disable an installed test speed recipe')
    os.environ['SATELLITE_TEST_ACCELERATION'] = '1' if enabled else '0'
    os.environ['SATELLITE_ACCELERATION'] = '1' if enabled else '0'


# Shared evaluation acceleration, formerly loaded from diagnostic scripts.
_NATIVE_MANIFEST = {'status': 'complete',
 'purpose': 'isolated candidate; original event and numeric operation order retained',
 'cython_version': '3.0.12',
 'wheel_sha256': 'dcdc3e5d4ce0e7a4af6903ed580833015641e968d18d528d8371e2435a34132c',
 'compiler_flags': ['/O2', '/fp:strict'],
 'infer_types': False,
 'targets': {'simulation': ['get_current_state',
                            'state_exchanger',
                            'valid_action_mask',
                            '_heartbeat_deadline',
                            'neighbor_is_alive',
                            'neighbor_has_resource_state',
                            '_refresh_transmission_features',
                            '_memory_allowed_action_indices',
                            '_projected_neighbor_memory_remain',
                            '_normalized_link_transmission_wait',
                            '_normalized_mean_transmission_wait'],
             'beta': ['update_adjacency_dict', 'update_adjacency_table'],
             'propagator': ['send_state', 'send_adjacency_table'],
             'runner': ['_speed_defer_routes',
                        '_speed_routing_table',
                        '_speed_root_routes',
                        '_speed_neighbor_distances'],
             'simpy_core': ['step', 'schedule', 'run'],
             'simpy_process': ['_resume']},
 'source_sha256': {'SatelliteNetworkSimulator_Computing': '15e7bf9e967c8caf4918a2733824c665911f5329bc81c93c840266c73424470e',
                   'SatelliteNetworkSimulator_Beta': 'd25539ed5e0f5e68870dc5e00cd5c15b4aa629499ab13377340fdd5457a694f9',
                   'training_runner': 'dca0dd02dbd3c3cc7f0423d349617eafedf3f37a09bd2a6d83e3ea2aee4cee35',
                   'simpy.core': '0274b36d9dda5c0901b8d765885705bbe76640e24da1bc63e679433e0f9c49dd',
                   'simpy.events': '7cba6e717092cee1b0d51ed33b83a2f51f055dc3327c894bf99122c3bfa16b4d'},
 'pyx_sha256': '229277d9fc0f32fa35e5ff2870383b1e29f90325fa04987b1419d6940b0f03f1',
 'binary_sha256': 'e34a385d5c9706e86c7264b92b164ba6474919814248d5bcd7093cb6d8a13ed6',
 'build_script_sha256': '8877bf8718440d849bd461580bc9bf8e2e23e0e5acfb88f120280a966e8d5601'}

_NATIVE_REVIEWED_SOURCE = {'simulation_sha256': 'fd5c7c5e73092ca476b4c001ddba9bd042f5836df538916fee089952202c1bbf',
 'simulation_ast_sha256': '6a2e92e63184e618648d05602796c0593a22113db8c675e66434eeb8ec84ee78',
 'runner_sha256': 'e9b6ec27c7e490cd6754126e16dc3d91be245da58f7110797c366bf858f30cb6',
 'runner_function_ast_sha256': {'_speed_defer_routes': '084c8a4a52bb02815f9757d27d4cc94a3dd17e826713551e06694b222f801f68',
                                '_speed_routing_table': '734df8df2435376985c47fb896b0eedd13b04b6f5a9bc0ee83d06b7b8665c40d',
                                '_speed_neighbor_distances': '3ad81091212004161652c8e2432237803de9719a89d2fc8a5880dcea39d3f1df',
                                '_speed_root_routes': '2b866a1a8b1bb87574d284d204d2a798ec73b7a81c92a69959813b564dcdc90a'}}

# oneweb_vector_merge_candidate.py (test runtime implementation)
class VersionedAdjacency(dict):
    def __init__(self, values):
        super().__init__(values)
        self.indices = None
        self.timestamps = None
        self.keys_in_order = None
        self.node_ids = None

    def prepare(self, node_ids):
        if self.node_ids is node_ids:
            return
        self.node_ids = node_ids
        self.timestamps = np.full(len(node_ids), -np.inf)
        self.present = np.zeros(len(node_ids), dtype=bool)
        for key, (_, timestamp) in self.items():
            self.timestamps[node_ids[key]] = timestamp
            self.present[node_ids[key]] = True
        self.keys_in_order = tuple(self)
        self.indices = np.asarray([node_ids[key] for key in self.keys_in_order], dtype=np.intp)

    def __setitem__(self, key, value):
        new_key = key not in self
        super().__setitem__(key, value)
        if self.node_ids is not None:
            self.timestamps[self.node_ids[key]] = value[1]
            self.present[self.node_ids[key]] = True
            if new_key:
                self.keys_in_order = tuple(self)
                self.indices = np.asarray([self.node_ids[item] for item in self.keys_in_order])


def _install_vector_merge():
    import SatelliteNetworkSimulator_Beta as beta
    original = beta.Satellite.update_adjacency_dict

    def get_table(self):
        return self.__dict__['adjacency_table']

    def set_table(self, value):
        self.__dict__['adjacency_table'] = VersionedAdjacency(value)

    def merge(self, incoming):
        local = self.adjacency_table
        if isinstance(local, VersionedAdjacency) and isinstance(incoming, VersionedAdjacency):
            propagator = self.propagator
            ids = getattr(propagator, '_diagnostic_node_ids', None)
            if ids is None:
                ids = {name: index for index, name in enumerate(propagator.satellites)}
                propagator._diagnostic_node_ids = ids
            local.prepare(ids)
            incoming.prepare(ids)
            # Read the live table at receive time, as the original code does.
            indices = incoming.indices
            # Original merge always accepts previously absent keys, even at -inf/NaN.
            changed = np.flatnonzero(~local.present[indices] |
                                    (incoming.timestamps[indices] > local.timestamps[indices]))
            if not len(changed):
                return False
            selected = {incoming.keys_in_order[index]: incoming[incoming.keys_in_order[index]]
                        for index in changed}
            return original(self, selected)
        return original(self, incoming)

    beta.Satellite.adjacency_table = property(get_table, set_table)
    beta.Satellite.update_adjacency_dict = merge

# large_decision_state_candidate.py (test runtime implementation)
def memory_scope(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if self.mode != 'New_DQN':
            return function(self, *args, **kwargs)
        previous = self._memory_allowed_action_indices
        owned = '_memory_allowed_action_indices' in vars(self)
        cached = {}

        def allowed(packet_size):
            if packet_size not in cached:
                cached[packet_size] = previous(packet_size)
            return cached[packet_size]

        self._memory_allowed_action_indices = allowed
        try:
            return function(self, *args, **kwargs)
        finally:
            if owned:
                self._memory_allowed_action_indices = previous
            else:
                del self._memory_allowed_action_indices
    return wrapped


def _install_decision_state(*, cache_memory=False):
    import test_config as selection
    import SatelliteNetworkSimulator_Computing as simulation
    original = selection.destination_selection.__wrapped__
    source = textwrap.dedent(inspect.getsource(original))
    before = '            rows = []'
    old = '                state = self.get_current_state(destination, hops, is_computed, mission_state)'
    assert source.count(before) == source.count(old) == 1
    if cache_memory:
        simulation._candidate_memory_scope = memory_scope
        decorator = '        @torch.no_grad()\n        def find_highest_score('
        assert source.count(decorator) == 1
        source = source.replace(decorator, '        @simulation._candidate_memory_scope\n' + decorator)
    source = source.replace(before, before + '\n            state_template = None\n            distance_slots = None')
    new = '''                if self.mode != 'New_DQN' or state_template is None:
                    state = self.get_current_state(destination, hops, is_computed, mission_state)
                    if self.mode == 'New_DQN':
                        state_template = state
                        live = {neighbor for neighbor in self.neighbors
                                if self.neighbor_has_resource_state(neighbor)}
                        slots = getattr(self, 'action_neighbors', self.neighbors)
                        distance_slots = [(index * simulation.NEIGHBOR_SLOT_DIM
                                           + simulation.NEIGHBOR_SLOT_DIM - 1, neighbor)
                                          for index, neighbor in enumerate(slots)
                                          if neighbor in live]
                else:
                    state = state_template.copy()
                    for index, neighbor in distance_slots:
                        distances = self.neighbor_hops[neighbor]
                        state[index] = (distances[destination] / self.max_hop
                                        if destination in distances else 2)'''
    source = source.replace(old, new)
    # Apply the prior batching optimization in this same source transformation.
    row = 'tensor = torch.as_tensor(policy_state, dtype=torch.float, device=self.device)'
    batch = 'states = torch.stack([row[4] for row in rows])'
    assert source.count(row) == source.count(batch) == 1
    source = source.replace(row, 'tensor = np.asarray(policy_state)')
    source = source.replace(batch, """states = (torch.stack([row[4] for row in rows]) if 'GNN' in self.mode
                      else torch.as_tensor(np.stack([row[4] for row in rows]),
                                           dtype=torch.float, device=self.device))""")
    namespace = dict(original.__globals__)
    exec(compile(source, '<isolated common decision state>', 'exec'), namespace)
    original.__code__ = namespace['destination_selection'].__wrapped__.__code__

# large_forward_versions_candidate.py (test runtime implementation)
def _install_forward_versions():
    import SatelliteNetworkSimulator_Beta as beta
    original = beta.Satellite.update_adjacency_table
    versioned = beta.Satellite.adjacency_table.fset.__globals__['VersionedAdjacency']

    def update(self, table):
        if not isinstance(table, versioned):
            # A fallback can change the dictionary without updating its dense view.
            self.__dict__.pop('_candidate_forward_versions', None)
            return original(self, table)
        if not self.update_adjacency_dict(table):
            return
        ids = self.propagator._diagnostic_node_ids
        table.prepare(ids)
        forwarded = getattr(self, '_adjacency_forwarded_versions', {})
        versions = getattr(self, '_candidate_forward_versions', None)
        if versions is None:
            versions = np.full(len(ids), -np.inf)
            for key, timestamp in forwarded.items():
                versions[ids[key]] = timestamp
            self._candidate_forward_versions = versions
        indices = table.indices
        changed = np.flatnonzero(table.timestamps[indices] > versions[indices])
        for index in changed:
            key = table.keys_in_order[index]
            forwarded[key] = table[key][1]
        versions[indices[changed]] = table.timestamps[indices[changed]]
        self._adjacency_forwarded_versions = forwarded
        self.build_routing_table()
        if len(changed):
            self.adjacency_table_exchanger()

    beta.Satellite.update_adjacency_table = update

# large_route_snapshot_candidate.py (test runtime implementation)
class TrackedNeighbors(list):
    def __init__(self, values=()):
        super().__init__(values)
        self.frozen = tuple(self)
        self.views = []

    def changed(self):
        self.frozen = tuple(self)
        alive = []
        for reference, key in self.views:
            table = reference()
            record = None if table is None else table.get(key)
            if record is not None and record[0] is self:
                table.frozen[key] = (self.frozen, record[1])
                alive.append((reference, key))
        self.views = alive


def mutation(method):
    def mutate(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.changed()
        return result
    return mutate


for name in ('append', 'remove', 'extend', 'insert', 'pop', 'clear', 'sort',
             'reverse', '__setitem__', '__delitem__', '__iadd__', '__imul__'):
    setattr(TrackedNeighbors, name, mutation(getattr(list, name)))


def _install_route_snapshot():
    import SatelliteNetworkSimulator_Beta as beta
    import SatelliteNetworkSimulator_Computing as simulation
    import training_runner as runner
    descriptor = beta.Satellite.adjacency_table
    base = descriptor.fset.__globals__['VersionedAdjacency']

    class LiveAdjacency(base):
        def __init__(self, values):
            super().__init__({})
            self.frozen = {}
            self.plain_rows = {}
            self.tracked_rows = {}
            for key, value in values.items():
                self[key] = value

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            neighbors, timestamp = value
            if isinstance(neighbors, TrackedNeighbors):
                self.frozen[key] = (neighbors.frozen, timestamp)
                if self.tracked_rows.get(key) is not neighbors:
                    neighbors.views.append((weakref.ref(self), key))
                    self.tracked_rows[key] = neighbors
                self.plain_rows.pop(key, None)
            else:
                # Unknown plain lists retain live-alias semantics via refresh.
                self.frozen[key] = (tuple(neighbors), timestamp)
                self.plain_rows[key] = value
                self.tracked_rows.pop(key, None)

        def snapshot(self):
            for key, (neighbors, timestamp) in self.plain_rows.items():
                self.frozen[key] = (tuple(neighbors), timestamp)
            return self.frozen.copy()

    def set_adjacency(self, value):
        self.__dict__['adjacency_table'] = LiveAdjacency(value)

    def get_neighbors(self):
        return self.__dict__['neighbors']

    def set_neighbors(self, value):
        self.__dict__['neighbors'] = (value if isinstance(value, TrackedNeighbors)
                                      else TrackedNeighbors(value))

    beta.Satellite.adjacency_table = property(descriptor.fget, set_adjacency)
    beta.Satellite.neighbors = property(get_neighbors, set_neighbors)
    original_extract = simulation.SatelliteNetworkSimulator_OnbardComputing.extract_adjacency_dict

    def extract(self):
        # These lists have just been allocated by the original extractor.
        return {key: (TrackedNeighbors(neighbors), timestamp)
                for key, (neighbors, timestamp) in original_extract(self).items()}

    simulation.SatelliteNetworkSimulator_OnbardComputing.extract_adjacency_dict = extract
    original_defer = runner._speed_defer_routes

    def defer(self):
        adjacency = getattr(self, 'adjacency_table', None)
        if not isinstance(adjacency, LiveAdjacency):
            return original_defer(self)
        self.__dict__['_speed_route_inputs'] = (
            adjacency.snapshot(), list(self.neighbors), dict(self._neighbor_route_since))
        self.__dict__['_speed_routes_dirty'] = True

    runner._speed_defer_routes = defer
    return LiveAdjacency

# large_route_journal_candidate.py (test runtime implementation)
class RouteSnapshot(Mapping):
    def __init__(self, parent, changes):
        self.parent = parent
        self.changes = changes
        self.materialized = None

    def resolve(self):
        if self.materialized is None:
            pending = []
            node = self
            while node is not None and node.materialized is None:
                pending.append(node)
                node = node.parent
            rows = {} if node is None else node.materialized.copy()
            for snapshot in reversed(pending):
                rows.update(snapshot.changes)
            self.materialized = rows
            self.parent = None
            self.changes = None
        return self.materialized

    def __getitem__(self, key):
        return self.resolve()[key]

    def __iter__(self):
        return iter(self.resolve())

    def __len__(self):
        return len(self.resolve())

    def items(self):
        return self.resolve().items()


class JournalRows(dict):
    def __init__(self, rows):
        super().__init__(rows)
        self.pending = dict(rows)
        self.head = None

    def __setitem__(self, key, value):
        if key not in self or self[key] != value:
            self.pending[key] = value
        super().__setitem__(key, value)

    def copy(self):
        if self.pending or self.head is None:
            self.head = RouteSnapshot(self.head, self.pending)
            self.pending = {}
        return self.head


def _install_route_journal():
    import SatelliteNetworkSimulator_Beta as beta
    live = inspect.getclosurevars(beta.Satellite.adjacency_table.fset).nonlocals['LiveAdjacency']
    original_init = live.__init__

    def initialize(self, values):
        original_init(self, values)
        self.frozen = JournalRows(self.frozen)

    live.__init__ = initialize

# large_merge_revision_candidate.py (test runtime implementation)
def cached_merge(original, table_type):
    def merge(self, incoming):
        local = self.adjacency_table
        if not isinstance(local, table_type) or not isinstance(incoming, table_type):
            return original(self, incoming)
        incoming_revision = getattr(incoming, '_merge_revision', 0)
        local_revision = getattr(local, '_merge_revision', 0)
        cache = getattr(local, '_merge_seen', None)
        if cache is None:
            cache = local._merge_seen = {}
        previous = cache.get(id(incoming))
        if (previous is not None and previous[0]() is incoming
                and previous[1:] == (incoming_revision, local_revision)):
            return False
        result = original(self, incoming)
        cache[id(incoming)] = (weakref.ref(incoming), incoming_revision,
                               getattr(local, '_merge_revision', 0))
        return result
    return merge


def _install_merge_revision():
    import SatelliteNetworkSimulator_Beta as beta
    # LiveAdjacency inherits the dense timestamp table installed by vector merge.
    import inspect
    live = inspect.getclosurevars(beta.Satellite.adjacency_table.fset).nonlocals['LiveAdjacency']
    table_type = live.__bases__[0]
    original_set = table_type.__setitem__

    def setitem(self, key, value):
        original_set(self, key, value)
        self._merge_revision = getattr(self, '_merge_revision', 0) + 1

    table_type.__setitem__ = setitem
    beta.Satellite.update_adjacency_dict = cached_merge(
        beta.Satellite.update_adjacency_dict, table_type)

# large_gc_freeze_candidate.py (test runtime implementation)
def _install_gc_freeze():
    import training_runner as runner
    original_build = runner.build_environment
    original_run = runner.run_training

    def build(config, agent, device, shuffle, **kwargs):
        if config['general']['phase'] != 'test':
            raise ValueError('GC freeze candidate is evaluation-only')
        env = original_build(config, agent, device, shuffle, **kwargs)
        original_reset = env.reset

        def reset(*args, **reset_kwargs):
            gc.unfreeze()
            result = original_reset(*args, **reset_kwargs)
            gc.collect()
            gc.freeze()
            return result

        env.reset = reset
        return env

    def run(config):
        if gc.get_freeze_count():
            raise RuntimeError('Diagnostic worker already contains frozen GC objects')
        try:
            return original_run(config)
        finally:
            gc.unfreeze()
            gc.collect()

    runner.build_environment = build
    runner.run_training = run

# large_gc_threshold_candidate.py (test runtime implementation)
def wrap_test_runtime(original):
    def run(config):
        if config['general']['phase'] != 'test':
            raise ValueError('GC threshold candidate is evaluation-only')
        thresholds = gc.get_threshold()
        try:
            gc.set_threshold(50000, 10, 1000)
            return original(config)
        finally:
            gc.set_threshold(*thresholds)
    return run


def _install_gc_threshold():
    import training_runner as runner
    runner.run_training = wrap_test_runtime(runner.run_training)

# large_joint_cpu_candidate.py (test runtime implementation)
def _install_joint_cpu():
    import torch
    import training_runner as runner
    original_build = runner.build_environment

    def build(config, agent, device, shuffle, **kwargs):
        if (config['general']['phase'] == 'test' and config['agent']['mode'] == 'New_DQN'
                and config['general'].get('select_mode') == 3
                and torch.device(device).type == 'cuda'):
            return runner._speed_build_cpu_environment(
                original_build, config, agent, device, shuffle, **kwargs)
        return original_build(config, agent, device, shuffle, **kwargs)

    runner.build_environment = build

# cpu_joint_tie_fallback_candidate.py (test runtime implementation)
_TEST_CPU_TIE_COUNTERS = {'cpu_calls': 0, 'cuda_tie_fallbacks': 0}


def checker(source):
    import torch
    def checked(output, state, mask):
        _TEST_CPU_TIE_COUNTERS['cpu_calls'] += 1
        scores = output.masked_fill(~mask.expand_as(output), float('-inf')).flatten()
        top = scores.topk(min(2, scores.numel())).values
        if top.numel() > 1 and float(top[0] - top[1]) < 0.0001:
            _TEST_CPU_TIE_COUNTERS['cuda_tie_fallbacks'] += 1
            # Re-evaluate the original complete batch, including identical
            # rows: different GEMM batch shapes can change roundoff/ties.
            return source(state.to('cuda'), mask.to('cuda')).cpu()
        return output
    return checked


def _install_cpu_tie():
    import training_runner as runner
    original = runner._speed_build_cpu_environment
    source = textwrap.dedent(inspect.getsource(original))
    before = '    cpu_net = copy.deepcopy(source).cpu()'
    inference = 'return scripted_net(state, mask)'
    assert source.count(before) == source.count(inference) == 1
    source = source.replace(before, before + '\n    checked = _tie_checker(source)')
    source = source.replace(inference, 'return checked(scripted_net(state, mask), state, mask)')
    namespace = dict(original.__globals__, _tie_checker=checker)
    exec(compile(source, '<isolated cpu joint tie fallback>', 'exec'), namespace)
    runner._speed_build_cpu_environment = namespace[original.__name__]

def _install_vector_forwarding():
    _install_vector_merge()
    import SatelliteNetworkSimulator_Beta as beta
    original = beta.Satellite.update_adjacency_table
    source = textwrap.dedent(inspect.getsource(original))
    old = "forwarded_versions.get(key, float('-inf'))"
    assert source.count(old) == 1
    source = source.replace(old, 'forwarded_versions.get(key, negative_infinity)')
    source = source.replace('    should_forward = False',
                            "    negative_infinity = float('-inf')\n    should_forward = False")
    namespace = dict(original.__globals__)
    exec(compile(source, '<isolated constant hoist>', 'exec'), namespace)
    beta.Satellite.update_adjacency_table = namespace['update_adjacency_table']

def _install_native_test_runtime(build):
    import importlib
    import SatelliteNetworkSimulator_Computing as simulation
    import SatelliteNetworkSimulator_Beta as beta
    import training_runner as runner
    manifest = _NATIVE_MANIFEST
    assert manifest.get('status', 'complete') == 'complete'
    binary, = build.glob('_satellite_native_hot*.pyd')
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest['binary_sha256']
    for module in (simulation, beta, runner):
        current = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if current == manifest['source_sha256'][module.__name__]:
            continue
        if module is simulation:
            assert current == _NATIVE_REVIEWED_SOURCE['simulation_sha256']
            tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
            assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == (
                _NATIVE_REVIEWED_SOURCE['simulation_ast_sha256'])
            assert 'forward_packet' not in manifest['targets']['simulation']
            continue
        assert module is runner, f'Native source changed: {module.__name__}'
        assert current == _NATIVE_REVIEWED_SOURCE['runner_sha256']
        functions = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                     for node in ast.parse(Path(runner.__file__).read_text(encoding='utf-8')).body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert all(functions[name] == expected for name, expected in
                   _NATIVE_REVIEWED_SOURCE['runner_function_ast_sha256'].items())
    sys.path.insert(0, str(build))
    native = importlib.import_module('_satellite_native_hot')
    beta.Satellite.update_adjacency_dict = native.update_adjacency_dict
    # Retain vector filtering, capturing the compiled original merge kernel.
    _install_vector_forwarding()
    for label, owner in (('simulation', simulation.Satellite_with_Computing),
                         ('beta', beta.Satellite), ('propagator', beta.Propagator),
                         ('runner', runner)):
        for name in manifest['targets'][label]:
            # The compiled state-exchange generator has crashed at the same
            # native instruction with either Python or compiled SimPy.
            # Retain the original generator and its Python-managed lifetime.
            if label == 'simulation' and name == 'state_exchanger':
                continue
            if label == 'beta' and name == 'update_adjacency_dict':
                continue
            setattr(owner, name, getattr(native, name))
    _install_decision_state(cache_memory=True)


def _install_test_speed(build):
    _install_native_test_runtime(build)
    _install_forward_versions()
    _install_route_snapshot()
    for flag, install in (
            ('JOURNAL', _install_route_journal), ('GC', _install_gc_freeze),
            ('GC_THRESHOLD', _install_gc_threshold), ('MERGE_REVISION', _install_merge_revision),
            ('JOINT_CPU', _install_joint_cpu)):
        if os.environ.get('SATELLITE_SPEED_' + flag) == '1':
            install()
    _install_cpu_tie()


def prepare_test_acceleration(config):
    """Install before entering destination-selection or hop/encoder contexts."""
    global _installed, _recipe
    if config['general']['phase'] != 'test':
        raise ValueError('The test speed recipe is evaluation-only')
    if not acceleration_enabled():
        configure_acceleration(False)
        config['test_acceleration'] = {'enabled': False}
        return
    if not _installed:
        configure_acceleration(True)
        for name in ('JOURNAL', 'GC', 'GC_THRESHOLD', 'MERGE_REVISION', 'JOINT_CPU'):
            os.environ['SATELLITE_SPEED_' + name] = '1'
        # GCSR repeatedly replaces cyclic NetworkX graphs; collect them normally.
        if config['general'].get('algorithm_name') == 'GCSR':
            os.environ['SATELLITE_SPEED_GC_THRESHOLD'] = '0'
        os.environ['SATELLITE_ACCELERATION_ROUTING'] = '1'
        # One binary per worker: all constellations may share the same pool.
        build = PROJECT_ROOT / 'training_process_data/v1/test/test/runtime/native'
        os.environ['SATELLITE_NATIVE_BUILD_DIR'] = str(build)
        _install_test_speed(build)
        binary, = build.glob('_satellite_native_hot*.pyd')
        files = [Path(__file__), binary]
        _recipe = dict(enabled=True, native_build=str(build), native_simpy=False,
                       python_state_exchanger=True,
                       cuda_tie_threshold=0.0001,
                       source_sha256={str(path.relative_to(PROJECT_ROOT)):
                                      hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
        _installed = True
    # GNN and PPO retain their policy implementations; only plain New_DQN
    # evaluation with select_mode=3 uses the additional CPU joint-policy path.
    config['test_acceleration'] = dict(_recipe, joint_cpu_eligible=(
        config['agent']['mode'] == 'New_DQN' and config['general'].get('select_mode') == 3))


def run_training(config):
    import training_runner as runner
    return runner.run_training(config)


if __name__ == '__main__':
    main()
