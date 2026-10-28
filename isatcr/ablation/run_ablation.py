"""Run one or all ISACR self-elimination aggregation ablations."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.training_entry import (
    TRAINING_SEEDS,
    build_config as build_shared_config,
    run_variant,
)


VARIANT_NAMES = {
    'direct_concat': 'DirectConcat',
    'gat': 'GAT',
    'graphsage': 'GraphSAGE',
    'jknet': 'JKNet',
    'no_self_feature': 'NoSelfFeature',
}


def build_config(encoder_name):
    """Backward-compatible access to the shared audited configuration."""
    return build_shared_config(encoder_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run iSatCR with a replacement graph aggregation module.',
    )
    parser.add_argument(
        '--encoder',
        required=True,
        choices=list(VARIANT_NAMES),
        help='Aggregation or feature-transfer ablation variant.',
    )
    parser.add_argument('--seed', required=True, type=int, choices=TRAINING_SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    run_variant(args.encoder, args.seed)


if __name__ == '__main__':
    main()
