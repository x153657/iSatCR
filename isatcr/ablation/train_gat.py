from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.training_entry import TRAINING_SEEDS, build_config as build_shared_config, main_for_encoder


CONFIG = build_shared_config('gat')


def seed_config(seed):
    return build_shared_config('gat', seed)


if __name__ == '__main__':
    main_for_encoder('gat')
