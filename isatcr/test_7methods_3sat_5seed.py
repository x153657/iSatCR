"""Seven methods on Telesat/OneWeb/Starlink, fixed load 160, fault 18%, fill 2.

ICS/GCSR run once per constellation; five learned methods use seeds 0..4.
ISACR imports completed coefficient-2 results instead of duplicating simulations.
Each new simulation has its own process so heuristic patches cannot leak.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import psutil

import test_5ix_5seed_5sat as reference
import test_4_5seed as baseline
from test_config import configure_acceleration, prepare_test_acceleration, run_training, destination_selection

# Use the same strictly reviewed source revisions for this fixed-model suite.
baseline.load_checkpoint_training_config = reference.load_checkpoint_training_config

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'training_process_data/v1/test/test/seven_methods_3sat_load160_fault18'
CONSTELLATIONS = ('Telesat', 'OneWeb', 'Starlink')
METHODS = ('ICS', 'GCSR', 'ISACR', 'D3QN+', 'D3QN', 'DDQN', 'PPO')
reference.configure_isacr_plus(True)


# Reviewed result compatibility and scheduling policy.
REVIEWED_VALIDATION = {'seven_completed_compatibility': {'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\TBO-SP\\single': {'identity_sha256': '0c4e969df425bfecabe63e9d315f063cfe90548b650a029e830d9c432ade1ff7',
                                                                                                                                          'raw_sha256': '56da67caf40dd1f775984fc07733a33390014ada76ce5593d38da773f0c981dd',
                                                                                                                                          'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                          'review': 'Exact '
                                                                                                                                                    'core '
                                                                                                                                                    'identity, '
                                                                                                                                                    'full300 '
                                                                                                                                                    'and '
                                                                                                                                                    'trace '
                                                                                                                                                    'summary '
                                                                                                                                                    'verified '
                                                                                                                                                    'for7-worker '
                                                                                                                                                    'scheduler-only '
                                                                                                                                                    'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\D3QN\\seed0': {'identity_sha256': '611487d955141d49b645fb16f76ff4014889b8b38c39748eafd11c96fd5a2454',
                                                                                                                                       'raw_sha256': '692f40e357c4a18300f1d9580d67821d5d3f86b028b9d4ae9cc8bae550aae8c6',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\D3QN\\seed1': {'identity_sha256': '643879a3a51eb183622623e3b6febceac1ae52f8d34cb904ba17592dbe6cdbc4',
                                                                                                                                       'raw_sha256': '573667c29f9f55fb7653f67f4780e18b36be53688030ede7b36051a53ed181ec',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\D3QN\\seed3': {'identity_sha256': 'fca6ef7a14f59e02a13f72482b3d5e1132492cb6d75432b8c795fbbeb89afe64',
                                                                                                                                       'raw_sha256': '99888245a8189bb2d9f1c062b92050d4bbc273c06cc3228416ca8dd4927ae28c',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\DDQN\\seed1': {'identity_sha256': 'e881b2a7edf88e80edb20ccecd7fa5932d5f8a24a948cc3dcc6adbc4beba33e7',
                                                                                                                                       'raw_sha256': 'cfff7f9cd1601923f844cd66cf36f3ee64d36c04ddb5b660fd1c4e2ea7c67125',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\DDQN\\seed4': {'identity_sha256': '14d9b414ddfd940cd2c873fc12d88798d7197153bcd6b736dda1a3288431a934',
                                                                                                                                       'raw_sha256': 'f1287a592b1ca24f8d00007a8dda46bbb1b24e32d2ef8711932f48d91bd6a018',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\PPO\\seed4': {'identity_sha256': 'f3f8a3bec4a6ad45e3d7560814e9537ad3b74aeb137589d4cbf1ffcf64483d20',
                                                                                                                                      'raw_sha256': 'c3b3c1cd1be744bd747b07dbbe17620a4d3736fec25597169ce6a79adbb8b9f5',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\D3QN\\seed2': {'identity_sha256': 'ea55b2d668e60f356bf0b6aaa145d9bf2da341ae698eb94aac60def950c2b16d',
                                                                                                                                       'raw_sha256': 'd19bb23376c61e030ea843bbe5bde3b3d8307890241ab5893fec66ddde69dd68',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\D3QN\\seed4': {'identity_sha256': 'ba0b187fa531c571b38110e99c6af59b4f3d0f2793225f9da19bf93007b237ad',
                                                                                                                                       'raw_sha256': '06afbe943130d78617b4ec47ebbb5c714e7f9f641babfa0e16a3bd136783eabf',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\DDQN\\seed0': {'identity_sha256': 'c10b8f749b3298921e9f8e9f820a1a4b5abf38ee56963117006f4ba96fa5ee44',
                                                                                                                                       'raw_sha256': 'c17ce8b71f431072ca5c13e12450ded0df37d6ef61769155bb8e1ecc58b7c14a',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\DDQN\\seed2': {'identity_sha256': 'fe5423c60ff2ddbfb61adb9fc4295111e8a38692e0a12d672b6988bfc6dd54c3',
                                                                                                                                       'raw_sha256': '538791ac302ce74f457d886ebfbcbcba271f7735e63890ea744b2ab255859657',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\DDQN\\seed3': {'identity_sha256': '8fba719805ee4ed730d656fd0b84aff039db65b09d1d2e69a09bc7206e3fc8ea',
                                                                                                                                       'raw_sha256': 'b981a6eec7140851a6c80ae5ebda402325a8cd22b82d3ca5efb591e3f24cac88',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\PPO\\seed0': {'identity_sha256': '5044e9709f1ab7e6d86e9343e0ffaf6e1101d10e952693b09e7e61766cc9daaa',
                                                                                                                                      'raw_sha256': '360e36c6a7b495705956c6499c3728bd0890e214a7ef378692e924dc9cc134f0',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\PPO\\seed1': {'identity_sha256': 'cc898772b43776d3777e8fba66a3556bf0ba1396649e504ad3d3c8f69b2afc6b',
                                                                                                                                      'raw_sha256': '610bd41dcf33422360f429498ae019c6af266e1a0b79b7856d72bb09f474ca53',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\PPO\\seed2': {'identity_sha256': 'df5c2976c4b7fef81d284b3a4504d1f48250cf324e2432e0c8c7e9acf351c166',
                                                                                                                                      'raw_sha256': '47ced4b2e6760933d44c904041edb343782be3b0af6ac0d7c580226d145bf711',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\PPO\\seed3': {'identity_sha256': '8df6d5e01b7a9e3d3d13c2065700d99cfe595ffea5320a6a3b07166a45b2e802',
                                                                                                                                      'raw_sha256': '9d5ec4a08e5c42fe540de3175a9e8073096c13897a39e74bb18ce05b4a744c16',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\Telesat\\ICS\\single': {'identity_sha256': 'a2de00c69bdb390ecc4bfd9840fb8ae0fe42b88513b6861b8bae1c46be79dccb',
                                                                                                                                       'raw_sha256': '249bdd8112d96a5ce8a09dbb3572ca9844f56c7743e0c2e0bd62fecd9a124b6b',
                                                                                                                                       'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                       'review': 'Exact '
                                                                                                                                                 'core '
                                                                                                                                                 'identity, '
                                                                                                                                                 'full300 '
                                                                                                                                                 'and '
                                                                                                                                                 'trace '
                                                                                                                                                 'summary '
                                                                                                                                                 'verified '
                                                                                                                                                 'for7-worker '
                                                                                                                                                 'scheduler-only '
                                                                                                                                                 'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\D3QN\\seed0': {'identity_sha256': '4e635c3525e379cf9e1c17c176a347514a565022b1af569886c51aba69abfb64',
                                                                                                                                      'raw_sha256': 'dd70832db9adcb6c30becd3dbc9c845043a0dbe0eed88e2135e975195465bcbf',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\D3QN\\seed1': {'identity_sha256': 'ceb1b13fb775e8d421435a7df89b13786f6d27356f18c370459364b3361254b0',
                                                                                                                                      'raw_sha256': 'e4ac52763cdd61fa975a88ad67a220ce5ba0d4fb0e9f9ab4e323eb86deea0ff3',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\D3QN\\seed2': {'identity_sha256': '601adce8e3febd95a53bb5595bfd5e16c33f5d25b381a46ede751b7212da667c',
                                                                                                                                      'raw_sha256': '10ea92e2fa065b0c86fc898e641b4ad050bb5992e15fdc7800b9295ae7a0d68f',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\D3QN\\seed3': {'identity_sha256': '28fbf1c141640a09c39474d8002b57274ed4ed81bc87d4d7e12a228c559f6693',
                                                                                                                                      'raw_sha256': '56853f8d1c386725178145dcbd895850dc7770692cafb4aa7cb3614be8ffc84c',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\D3QN\\seed4': {'identity_sha256': 'b7e3b082dbd4e13ad14f0398d6351e71e11a571f4b434c75e36870b9cec7f4f8',
                                                                                                                                      'raw_sha256': 'fe1980272eeeb0b6982b29f546993e17380572316a6bae2fa398411fe0a09d31',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\DDQN\\seed0': {'identity_sha256': '73bad48a6f909283f8b6562d40a85173576e01a41e5a3444ae2cb6f9582bb704',
                                                                                                                                      'raw_sha256': 'bdb36002bdcce3477be3e6860292c24409295abfbfe6e6f31d7d8919c05260cf',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\DDQN\\seed1': {'identity_sha256': '7e38edd37d10de6944f203e4b9972e94d12a70388c7542cc17ddc58ffa785cce',
                                                                                                                                      'raw_sha256': 'fb814129de23698abbd2ccb8bb40af6afc450f1f3dbb2c82ef41cdffa6d2daee',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\DDQN\\seed2': {'identity_sha256': '6f75d8a871d76cc76433cf09ea15c70a83bc6ebacf3fcd35f782a0797ee90a9b',
                                                                                                                                      'raw_sha256': '34c5bc68d7a795f6deb998b2ab59ee79618d756937b204be657162285481a1f6',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\DDQN\\seed3': {'identity_sha256': '8ead41ca3b08929eb8bd46a9832130d473ac00231d26b6baaf6fbd293efb39f3',
                                                                                                                                      'raw_sha256': '005a9bea636fdbe2e788c8077e3345d125ab80e751235878e3ed64ad2d8ec7e1',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Exact '
                                                                                                                                                'core '
                                                                                                                                                'identity, '
                                                                                                                                                'full300 '
                                                                                                                                                'and '
                                                                                                                                                'trace '
                                                                                                                                                'summary '
                                                                                                                                                'verified '
                                                                                                                                                'for7-worker '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'handoff'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\DDQN\\seed4': {'identity_sha256': '3401a8727db06e93a0d1080470dd55189efbc42c33cdb29853e6487263d5edc7',
                                                                                                                                      'raw_sha256': '800e386ba109e7f3f3b753f580ec1459b63e5643ca247baba5cb5693f7b62813',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Preserved '
                                                                                                                                                'exact '
                                                                                                                                                'live '
                                                                                                                                                'case '
                                                                                                                                                'across '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'repair; '
                                                                                                                                                'exit0 '
                                                                                                                                                'and '
                                                                                                                                                'fixed '
                                                                                                                                                'trace '
                                                                                                                                                'verified'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\PPO\\seed0': {'identity_sha256': '1c653ff077b3432c7d37e5bcdd62731ce46a7b8679506a623e6463139249ac4d',
                                                                                                                                     'raw_sha256': '654c25689fac5971d6cfdc0ee542c5a11494a65ea5ae741f39dc134521ffa36d',
                                                                                                                                     'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                     'review': 'Preserved '
                                                                                                                                               'exact '
                                                                                                                                               'live '
                                                                                                                                               'case '
                                                                                                                                               'across '
                                                                                                                                               'scheduler-only '
                                                                                                                                               'repair; '
                                                                                                                                               'exit0 '
                                                                                                                                               'and '
                                                                                                                                               'fixed '
                                                                                                                                               'trace '
                                                                                                                                               'verified'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\PPO\\seed1': {'identity_sha256': 'a6064d03a7e8dbc9c2312859e43947b8f5038e35c8ad8b8f3fce88f701a6c69b',
                                                                                                                                     'raw_sha256': 'a5609d7327fbc8d6ef8e5d2fb2f726f428c1503df9a15fda77fb8f7ed317c3b0',
                                                                                                                                     'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                     'review': 'Preserved '
                                                                                                                                               'exact '
                                                                                                                                               'live '
                                                                                                                                               'case '
                                                                                                                                               'across '
                                                                                                                                               'scheduler-only '
                                                                                                                                               'repair; '
                                                                                                                                               'exit0 '
                                                                                                                                               'and '
                                                                                                                                               'fixed '
                                                                                                                                               'trace '
                                                                                                                                               'verified'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\PPO\\seed2': {'identity_sha256': '98af5b89262195025c2a7359f28090ed5a0554fa0ab7559de19a78aeceaacdc1',
                                                                                                                                     'raw_sha256': '795bd6c8d0f77dba291390853c9e9ebb1df1da19a8ddb73ab7fc7e2abb09b640',
                                                                                                                                     'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                     'review': 'Preserved '
                                                                                                                                               'exact '
                                                                                                                                               'live '
                                                                                                                                               'case '
                                                                                                                                               'across '
                                                                                                                                               'scheduler-only '
                                                                                                                                               'repair; '
                                                                                                                                               'exit0 '
                                                                                                                                               'and '
                                                                                                                                               'fixed '
                                                                                                                                               'trace '
                                                                                                                                               'verified'},
                                   'training_process_data\\v1\\test\\test\\seven_methods_3sat_load160_fault18\\OneWeb\\ICS\\single': {'identity_sha256': 'c0af79de23009ed50e235f3a870525922cbb5acf5711ce50912d55839c432c06',
                                                                                                                                      'raw_sha256': 'b23f65d8b4fb678a7a264a3790c496cf3f9431cab19fadfb90fd7a1a573acaa4',
                                                                                                                                      'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065',
                                                                                                                                      'review': 'Preserved '
                                                                                                                                                'exact '
                                                                                                                                                'live '
                                                                                                                                                'case '
                                                                                                                                                'across '
                                                                                                                                                'scheduler-only '
                                                                                                                                                'repair; '
                                                                                                                                                'exit0 '
                                                                                                                                                'and '
                                                                                                                                                'fixed '
                                                                                                                                                'trace '
                                                                                                                                                'verified'}},
 'seven_schedule_policy': {'epoch': 1792733231.4366105,
                           'authorization': 'User explicitly increases to7 concurrent workers; preserve '
                                            'current cases and user interrupted OneWeb TBO-SP',
                           'wait_for_ablation': False,
                           'workers': 7,
                           'interrupted_children': [{'pid': 51672,
                                                     'create_time': 1792715970.2232912,
                                                     'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                                                 '-B',
                                                                 '-u',
                                                                 '-X',
                                                                 'faulthandler',
                                                                 'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                                                 '--case',
                                                                 'Telesat',
                                                                 'ICS',
                                                                 'single',
                                                                 '--acceleration',
                                                                 'on']},
                                                    {'pid': 53928,
                                                     'create_time': 1792716420.801847,
                                                     'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                                                 '-B',
                                                                 '-u',
                                                                 '-X',
                                                                 'faulthandler',
                                                                 'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                                                 '--case',
                                                                 'OneWeb',
                                                                 'ICS',
                                                                 'single',
                                                                 '--acceleration',
                                                                 'on']},
                                                    {'pid': 52248,
                                                     'create_time': 1792716691.5053806,
                                                     'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                                                 '-B',
                                                                 '-u',
                                                                 '-X',
                                                                 'faulthandler',
                                                                 'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                                                 '--case',
                                                                 'OneWeb',
                                                                 'D3QN',
                                                                 '0',
                                                                 '--acceleration',
                                                                 'on']},
                                                    {'pid': 52940,
                                                     'create_time': 1792717570.488102,
                                                     'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                                                 '-B',
                                                                 '-u',
                                                                 '-X',
                                                                 'faulthandler',
                                                                 'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                                                 '--case',
                                                                 'OneWeb',
                                                                 'D3QN',
                                                                 '2',
                                                                 '--acceleration',
                                                                 'on']}],
                           'supersedes_memory_pause': True,
                           'preserve_completed': True,
                           'excluded_cases': []},
 'seven_orphan_workers': [{'pid': 10576,
                           'create_time': 1792742410.6270835,
                           'case': ['OneWeb', 'PPO', 1],
                           'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                       '-B',
                                       '-u',
                                       '-X',
                                       'faulthandler',
                                       'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                       '--case',
                                       'OneWeb',
                                       'PPO',
                                       '1',
                                       '--acceleration',
                                       'on'],
                           'expected_identity': {'config_sha256': '57b4f347076598514086d35ab13d9a756cf9dc115caaa60e1babf6f2369a61e9',
                                                 'checkpoint_sha256': {'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed1_train_reward_best_reward0p766727_delay4p665_loss0p000000_hops8p978\\actor.pth': '0cd7c29d31cd65be12584aa3d8fd66cd671ed7e0a27abb682680f7d021bd2b81',
                                                                       'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed1_train_reward_best_reward0p766727_delay4p665_loss0p000000_hops8p978\\critic.pth': 'b44bee5464d91a00017de03f67015b903ab42353ea553bdf73169e5998e0710b',
                                                                       'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed1_train_reward_best_reward0p766727_delay4p665_loss0p000000_hops8p978.training.json': 'a2ed8d89f7d30259af8a7f5028f23b762104b52cf3fddee7802659b6794d02bb'},
                                                 'sources': {'test_7methods_3sat_5seed.py': '0b578e3753085af53a0de3832edadd12fed30b5d287e87d15cbfa1373510ab07',
                                                             'test_ICS.py': '64b508bd8d8e58450d2237e5906f885963a89756e3dc9cedc7ac2396f8fbd92f',
                                                             'test_GSJF.py': '330a2694f159423dc1c9a4d45667fe41c91593e172b4dc7af98e398312714798',
                                                             'test_TBO-SP.py': '1f4508613b950921f197ad2bb037ee79c878a2e95129d972d4c67ba7cd2d7dc2',
                                                             'test_config.py': 'd24214aceafa2ca026a064a087c59fee9fc763f63584d4b52b69ea724a4bf81d',
                                                             'SatelliteNetworkSimulator_Computing.py': '15e7bf9e967c8caf4918a2733824c665911f5329bc81c93c840266c73424470e',
                                                             'RL_environment_for_computing.py': '57da9bd9d0a523db40922b8f5c97758e4c0ac299ab9a662b1f9bbd6f091299d1',
                                                             'Base_Agents.py': '0d41251ed8da137bc3aba8786fdd3021f7821f017f057dbfd80618be66babd0e',
                                                             'training_runner.py': 'e9b6ec27c7e490cd6754126e16dc3d91be245da58f7110797c366bf858f30cb6'}},
                           'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065'},
                          {'pid': 17404,
                           'create_time': 1792743750.0062587,
                           'case': ['OneWeb', 'PPO', 2],
                           'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                       '-B',
                                       '-u',
                                       '-X',
                                       'faulthandler',
                                       'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                       '--case',
                                       'OneWeb',
                                       'PPO',
                                       '2',
                                       '--acceleration',
                                       'on'],
                           'expected_identity': {'config_sha256': 'c1533cb40e52c031f74468574035f0c4d02f2f15a29495afb2da884a551bb275',
                                                 'checkpoint_sha256': {'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed2_train_reward_best_reward0p699330_delay5p277_loss0p014815_hops8p496\\actor.pth': '4471f5d8ca119b31bb20fd38c50a666a319b6af5474ea8e43ec55b739fae1d78',
                                                                       'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed2_train_reward_best_reward0p699330_delay5p277_loss0p014815_hops8p496\\critic.pth': '14688a634588b5ff1bf2c0fe518981b90048fd0b9bb2ba366a048137a5150fdd',
                                                                       'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed2_train_reward_best_reward0p699330_delay5p277_loss0p014815_hops8p496.training.json': 'cee924ca8ebafa4346c35732aa3db9c1e72fe444ae558c506c5489a1e7497301'},
                                                 'sources': {'test_7methods_3sat_5seed.py': '0b578e3753085af53a0de3832edadd12fed30b5d287e87d15cbfa1373510ab07',
                                                             'test_ICS.py': '64b508bd8d8e58450d2237e5906f885963a89756e3dc9cedc7ac2396f8fbd92f',
                                                             'test_GSJF.py': '330a2694f159423dc1c9a4d45667fe41c91593e172b4dc7af98e398312714798',
                                                             'test_TBO-SP.py': '1f4508613b950921f197ad2bb037ee79c878a2e95129d972d4c67ba7cd2d7dc2',
                                                             'test_config.py': 'd24214aceafa2ca026a064a087c59fee9fc763f63584d4b52b69ea724a4bf81d',
                                                             'SatelliteNetworkSimulator_Computing.py': '15e7bf9e967c8caf4918a2733824c665911f5329bc81c93c840266c73424470e',
                                                             'RL_environment_for_computing.py': '57da9bd9d0a523db40922b8f5c97758e4c0ac299ab9a662b1f9bbd6f091299d1',
                                                             'Base_Agents.py': '0d41251ed8da137bc3aba8786fdd3021f7821f017f057dbfd80618be66babd0e',
                                                             'training_runner.py': 'e9b6ec27c7e490cd6754126e16dc3d91be245da58f7110797c366bf858f30cb6'}},
                           'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065'},
                          {'pid': 40124,
                           'create_time': 1792741209.173437,
                           'case': ['OneWeb', 'PPO', 0],
                           'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                       '-B',
                                       '-u',
                                       '-X',
                                       'faulthandler',
                                       'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                       '--case',
                                       'OneWeb',
                                       'PPO',
                                       '0',
                                       '--acceleration',
                                       'on'],
                           'expected_identity': {'config_sha256': '14c8313950ee7443ec8745c4cc344c752e487fbdbc72a262995ec188bef8ac92',
                                                 'checkpoint_sha256': {'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed0_train_reward_best_reward0p755981_delay4p846_loss0p000000_hops8p370\\actor.pth': 'dfb80f0cd99124fd04227f21ad64add2c7895c120a89df4bdb65e6e8a35b42bd',
                                                                       'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed0_train_reward_best_reward0p755981_delay4p846_loss0p000000_hops8p370\\critic.pth': 'f78651d87647d776319ca7d2edd60437cdeebddac3ec7addab0ccb4ec8ca82d7',
                                                                       'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\ppo\\PPO_seed0_train_reward_best_reward0p755981_delay4p846_loss0p000000_hops8p370.training.json': '8dc0feb0db9f04e8a0280a30aed63c29cd50b89c556f82ee06a1563110265cc0'},
                                                 'sources': {'test_7methods_3sat_5seed.py': '0b578e3753085af53a0de3832edadd12fed30b5d287e87d15cbfa1373510ab07',
                                                             'test_ICS.py': '64b508bd8d8e58450d2237e5906f885963a89756e3dc9cedc7ac2396f8fbd92f',
                                                             'test_GSJF.py': '330a2694f159423dc1c9a4d45667fe41c91593e172b4dc7af98e398312714798',
                                                             'test_TBO-SP.py': '1f4508613b950921f197ad2bb037ee79c878a2e95129d972d4c67ba7cd2d7dc2',
                                                             'test_config.py': 'd24214aceafa2ca026a064a087c59fee9fc763f63584d4b52b69ea724a4bf81d',
                                                             'SatelliteNetworkSimulator_Computing.py': '15e7bf9e967c8caf4918a2733824c665911f5329bc81c93c840266c73424470e',
                                                             'RL_environment_for_computing.py': '57da9bd9d0a523db40922b8f5c97758e4c0ac299ab9a662b1f9bbd6f091299d1',
                                                             'Base_Agents.py': '0d41251ed8da137bc3aba8786fdd3021f7821f017f057dbfd80618be66babd0e',
                                                             'training_runner.py': 'e9b6ec27c7e490cd6754126e16dc3d91be245da58f7110797c366bf858f30cb6'}},
                           'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065'},
                          {'pid': 46940,
                           'create_time': 1792734478.434929,
                           'case': ['OneWeb', 'ICS', None],
                           'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                       '-B',
                                       '-u',
                                       '-X',
                                       'faulthandler',
                                       'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                       '--case',
                                       'OneWeb',
                                       'ICS',
                                       'single',
                                       '--acceleration',
                                       'on'],
                           'expected_identity': {'config_sha256': '714bc0f681520839e5b4c67003a5cf3db0890dda34aea15f6e4a39ae7ec5ff55',
                                                 'checkpoint_sha256': None,
                                                 'sources': {'test_7methods_3sat_5seed.py': '0b578e3753085af53a0de3832edadd12fed30b5d287e87d15cbfa1373510ab07',
                                                             'test_ICS.py': '64b508bd8d8e58450d2237e5906f885963a89756e3dc9cedc7ac2396f8fbd92f',
                                                             'test_GSJF.py': '330a2694f159423dc1c9a4d45667fe41c91593e172b4dc7af98e398312714798',
                                                             'test_TBO-SP.py': '1f4508613b950921f197ad2bb037ee79c878a2e95129d972d4c67ba7cd2d7dc2',
                                                             'test_config.py': 'd24214aceafa2ca026a064a087c59fee9fc763f63584d4b52b69ea724a4bf81d',
                                                             'SatelliteNetworkSimulator_Computing.py': '15e7bf9e967c8caf4918a2733824c665911f5329bc81c93c840266c73424470e',
                                                             'RL_environment_for_computing.py': '57da9bd9d0a523db40922b8f5c97758e4c0ac299ab9a662b1f9bbd6f091299d1',
                                                             'Base_Agents.py': '0d41251ed8da137bc3aba8786fdd3021f7821f017f057dbfd80618be66babd0e',
                                                             'training_runner.py': 'e9b6ec27c7e490cd6754126e16dc3d91be245da58f7110797c366bf858f30cb6'}},
                           'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065'},
                          {'pid': 48452,
                           'create_time': 1792741101.4772947,
                           'case': ['OneWeb', 'DDQN', 4],
                           'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                       '-B',
                                       '-u',
                                       '-X',
                                       'faulthandler',
                                       'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                       '--case',
                                       'OneWeb',
                                       'DDQN',
                                       '4',
                                       '--acceleration',
                                       'on'],
                           'expected_identity': {'config_sha256': '91da6c267b73087e7a24c5b6e0e97d9042796070dd8f7a0ca3e5b4c7822c5583',
                                                 'checkpoint_sha256': {'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\DDQN\\DDQN_seed4_train_reward_best_reward0p788208_delay4p236_loss0p000000_hops10p390.pth': '01efb694bd39b5374151f6cc97e029d965b7601a8cc250ccfb986e72fe34b1bb',
                                                                       'E:\\Satellite_simulation_golden1\\training_process_data\\v1\\model\\DDQN\\DDQN_seed4_train_reward_best_reward0p788208_delay4p236_loss0p000000_hops10p390.pth.training.json': '4953131a0be90167ef6b24b3e23eef1df523620c9d36e018f403e0a0eeb01d17'},
                                                 'sources': {'test_7methods_3sat_5seed.py': '0b578e3753085af53a0de3832edadd12fed30b5d287e87d15cbfa1373510ab07',
                                                             'test_ICS.py': '64b508bd8d8e58450d2237e5906f885963a89756e3dc9cedc7ac2396f8fbd92f',
                                                             'test_GSJF.py': '330a2694f159423dc1c9a4d45667fe41c91593e172b4dc7af98e398312714798',
                                                             'test_TBO-SP.py': '1f4508613b950921f197ad2bb037ee79c878a2e95129d972d4c67ba7cd2d7dc2',
                                                             'test_config.py': 'd24214aceafa2ca026a064a087c59fee9fc763f63584d4b52b69ea724a4bf81d',
                                                             'SatelliteNetworkSimulator_Computing.py': '15e7bf9e967c8caf4918a2733824c665911f5329bc81c93c840266c73424470e',
                                                             'RL_environment_for_computing.py': '57da9bd9d0a523db40922b8f5c97758e4c0ac299ab9a662b1f9bbd6f091299d1',
                                                             'Base_Agents.py': '0d41251ed8da137bc3aba8786fdd3021f7821f017f057dbfd80618be66babd0e',
                                                             'training_runner.py': 'e9b6ec27c7e490cd6754126e16dc3d91be245da58f7110797c366bf858f30cb6'}},
                           'runtime_ast_sha256': '3af3850a6447aeea4d3c7814d978ae1e7ff031f626f2db94dbb9b3afd7e8f065'}],
 'shared_seven_methods': {'pid': 20376,
                          'create_time': 1792743960.133757,
                          'command': ['D:\\anaconda\\envs\\xbb\\python.exe',
                                      '-B',
                                      '-u',
                                      '-X',
                                      'faulthandler',
                                      'E:\\Satellite_simulation_golden1\\test_7methods_3sat_5seed.py',
                                      '--workers',
                                      '15',
                                      '--acceleration',
                                      'on'],
                          'console': 'E:\\Satellite_simulation_golden1\\diagnostics\\ablation_remaining17_20261022_182539\\seven_methods15_console.txt',
                          'workers': 7,
                          'authorization': 'User increases7 parallel; existing manager adopted5 live cases '
                                           'and fills2',
                          'epoch': 1792743969.5830574}}


def directory(constellation, method, seed):
    return OUTPUT / constellation / method / ('single' if seed is None else f'seed{seed}')


def build_config(constellation, method, seed, raw):
    data = reference.CONFIGS[constellation]
    if method == 'ICS':
        import test_ICS as ics
        config = copy.deepcopy(ics.FIXED_CONFIG)
    elif method == 'GCSR':
        import test_GCSR as gcsr
        config = gcsr.build_config('load', data['load'], raw, scope='one_hop')
    elif method == 'ISACR':
        checkpoint = reference.select_model(seed, 2.0)
        config = reference.build_case_config(constellation, checkpoint, 2.0, raw)
    else:
        checkpoint, _, reason = baseline.find_ready_model(method, seed)
        if checkpoint is None:
            raise RuntimeError(f'{method}/seed{seed}: {reason}')
        config = baseline.build_test_config_base(method, data['load'], checkpoint)
        config['agent']['model_path'] = str(checkpoint)
    config['general'].update(data['common']['general'])
    config['general'].update(algorithm_name=method, random_seed=reference.TEST_SEED,
                             scenario_seed=reference.TEST_SEED, print_cycle=1,
                             show_test_summary=True, model_saving_enabled=False)
    if method in ('ISACR', 'D3QN+'):
        config['general']['select_mode'] = 3
    config['environment'].update(data['common']['environment'])
    config['environment'].update(data['load'])
    config['environment'].update(
        edge_p=0.18, edge_q=1.0 - 0.18, node_p=0.0, node_q=0.0, markov_mode=True,
        fault_fill_multiplier=2.0, fault_compute_double_mean=True,
        require_computed_before_downlink=False,
        visualize=False, save_log=False,
        save_training_data=str(raw),
        workload_trace_path=str(data['workload_path']),
        workload_trace_sha256=data['workload']['sha256'],
        workload_trace_task_count=data['workload']['task_count'],
        fault_trace_path=str(data['fault_trace_path']),
        fault_trace_sha256=data['fault_trace']['sha256'],
        fault_trace_event_count=data['fault_trace']['event_count'])
    config.pop('test_contract', None)
    config.pop('coefficient_test', None)
    config['batch_case'] = dict(constellation=constellation, method=method, training_seed=seed,
                                strategy=method)
    return config


def identity(config):
    value = copy.deepcopy(config)
    value['environment'].pop('save_training_data', None)
    value.pop('test_acceleration', None)
    checkpoint = value['agent'].get('model_path')
    return dict(config_sha256=hashlib.sha256(json.dumps(value, sort_keys=True,
                     default=str).encode()).hexdigest(),
                checkpoint_sha256=baseline.checkpoint_hashes(Path(checkpoint)) if checkpoint else None,
                sources={name: reference.digest(ROOT / name) for name in (
                    Path(__file__).name, 'test_ICS.py', 'test_GCSR.py', 'test_config.py',
                    'SatelliteNetworkSimulator_Computing.py', 'RL_environment_for_computing.py',
                    'Base_Agents.py', 'training_runner.py')})


def complete(folder, expected):
    try:
        saved = reference.read_json(folder / 'identity.json')
        if saved != expected:
            # Preserve individually audited results from scheduler-only revisions.
            validation = copy.deepcopy(REVIEWED_VALIDATION)
            reviewed = validation.get('seven_completed_compatibility', {}).get(str(folder.relative_to(ROOT)), {})
            runtime_names = ('build_config', 'identity', 'verified_copy', 'summarize', 'import_isacr', 'execute')
            tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
            runtime = hashlib.sha256(''.join(ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in runtime_names).encode()).hexdigest()
            if (reviewed.get('identity_sha256') != reference.digest(folder / 'identity.json')
                    or reviewed.get('raw_sha256') != reference.digest(folder / 'result.txt')
                    or reviewed.get('runtime_ast_sha256') != runtime):
                return False
            saved = copy.deepcopy(saved)
            saved['sources'][Path(__file__).name] = expected['sources'][Path(__file__).name]
        return (reference.last_step(folder / 'result.txt') == reference.EXPECTED_STEP
                and reference.read_json(folder / 'process_result.json')['exit_code'] == 0
                and saved == expected
                and (folder / 'summary.json').is_file())
    except (OSError, ValueError, KeyError):
        return False


def verified_copy(folder, config):
    provenance = folder / 'copy_provenance.json'
    case = config['batch_case']
    if case['method'] not in ('ISACR', 'D3QN+') or not provenance.is_file():
        return False
    copied = reference.read_json(provenance)
    source = Path(copied['copied_from'])
    raw = folder / 'result.txt'
    expected_hash = copied.get('source_result_sha256') or copied['files_sha256']['result.txt']
    if reference.digest(raw) != expected_hash or reference.digest(source / 'result.txt') != expected_hash:
        raise ValueError(f'Copied result changed: {folder}')
    saved = reference.read_json(folder / 'config.json')
    for section in ('general', 'agent', 'environment'):
        left, right = copy.deepcopy(saved[section]), copy.deepcopy(config[section])
        if section == 'environment':
            left.pop('save_training_data', None); right.pop('save_training_data', None)
        if case['method'] == 'ISACR' and section == 'general':
            if left.get('algorithm_name') in ('ISACR', 'ISACR_self_elimination'):
                left['algorithm_name'] = 'ISACR'
        if section == 'agent' and left.get('model_path') != right.get('model_path'):
            if reference.digest(Path(left['model_path'])) != reference.digest(Path(right['model_path'])):
                raise ValueError(f'Copied result checkpoint differs: {folder}')
            left['model_path'] = right['model_path']
        if left != right:
            raise ValueError(f'Copied result settings differ: {folder}/{section}')
    if reference.read_json(folder / 'process_result.json').get('exit_code') != 0:
        raise ValueError(f'Copied result did not finish: {folder}')
    source_config = reference.read_json(source / 'config.json')
    recorded = source_config.get('test_contract', {}).get('source_sha256', {})
    if not recorded:
        recorded = source_config.get('test_contract', {}).get('simulation_source_sha256', {})
    revisions = {
        'Base_Agents.py': ('ee4cd7eaa9da9eaf520851d51aa15e93d3583c7a51110a45e8c6f26469163f78',
                           '0d41251ed8da137bc3aba8786fdd3021f7821f017f057dbfd80618be66babd0e'),
        'training_runner.py': ('dca0dd02dbd3c3cc7f0423d349617eafedf3f37a09bd2a6d83e3ea2aee4cee35',
                              'e9b6ec27c7e490cd6754126e16dc3d91be245da58f7110797c366bf858f30cb6'),
    }
    for name in ('SatelliteNetworkSimulator_Computing.py', 'SatelliteNetworkSimulator_Beta.py',
                 'RL_environment_for_computing.py', 'Base_Agents.py', 'training_runner.py'):
        actual = reference.digest(ROOT / name)
        if recorded.get(name) != actual and revisions.get(name) != (recorded.get(name), actual):
            raise ValueError(f'Copied result simulator differs: {folder}/{name}')
    summarize(case['constellation'], case['method'], case['training_seed'], folder, write=False)
    return True


def summarize(constellation, method, seed, folder, write=True):
    data = reference.CONFIGS[constellation]
    raw = folder / 'result.txt'
    text = raw.read_text(encoding='utf-8')
    if (reference.last_step(raw) != reference.EXPECTED_STEP
            or 'Fixed test trace replay: verified' not in text):
        raise ValueError(f'Incomplete or unverified trace: {raw}')
    row = reference.parse_test_result(raw, method, data['load'], reference.DURATION,
                                      satellite_count=data['satellite_count'])
    for key, entry in (('workload', data['workload']), ('fault', data['fault_trace'])):
        if row.get(f'{key}_trace_sha256') != entry['sha256']:
            raise ValueError(f'{key} trace mismatch: {raw}')
    row.update(constellation=constellation, algorithm=method, training_seed=seed,
               fault_fill_multiplier=2.0, edge_p=0.18, edge_q=0.82,
               raw_result_file=str(raw))
    if write:
        reference.write_json(folder / 'summary.json', row)
    return row


def import_isacr(constellation, seed, config, dry_run=False):
    source = reference.case_directory(constellation, 2.0, seed)
    saved = reference.read_json(source / 'config.json')
    process = reference.read_json(source / 'process_result.json')
    if process.get('exit_code') != 0 or saved['coefficient_test']['training_seed'] != seed:
        raise ValueError('Coefficient case is not complete')
    # Compare actual simulation settings; runner/config defaults can change
    # without changing the already-recorded coefficient-2 experiment.
    for section in ('general', 'agent', 'environment'):
        left, right = copy.deepcopy(saved[section]), copy.deepcopy(config[section])
        if section == 'general':
            left.pop('algorithm_name', None); right.pop('algorithm_name', None)
        if section == 'environment':
            left.pop('save_training_data', None); right.pop('save_training_data', None)
        if left != right:
            raise ValueError(f'ISACR reuse settings differ: {section}')
    if saved['test_contract'].get('isacr_plus') is not True:
        raise ValueError('ISACR+ required')
    checkpoint = Path(config['agent']['model_path'])
    if process.get('model_sha256') != reference.digest(checkpoint):
        raise ValueError('ISACR checkpoint changed')
    for name, digest in saved['test_contract']['simulation_source_sha256'].items():
        if name in ('SatelliteNetworkSimulator_Computing.py', 'SatelliteNetworkSimulator_Beta.py',
                    'RL_environment_for_computing.py', 'Base_Agents.py', 'training_runner.py'):
            if reference.digest(ROOT / name) != digest:
                raise ValueError(f'ISACR simulator changed: {name}')
    # Verify the source before copying and keep provenance beside the copy.
    summarize(constellation, 'ISACR', seed, source, write=False)
    if dry_run:
        return
    folder = directory(constellation, 'ISACR', seed)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'result.txt', folder / 'result.txt')
    reference.write_json(folder / 'config.json', config)
    reference.write_json(folder / 'identity.json', identity(config))
    reference.write_json(folder / 'process_result.json', dict(exit_code=0, reused_from=str(source)))
    summarize(constellation, 'ISACR', seed, folder)


def execute(constellation, method, seed, acceleration):
    configure_acceleration(acceleration == 'on')
    folder = directory(constellation, method, seed)
    folder.mkdir(parents=True, exist_ok=True)
    running = folder / 'result.running.txt'
    config = build_config(constellation, method, seed, running)
    expected = identity(config)
    if complete(folder, expected) or verified_copy(folder, config):
        return
    from training_runner import archive_test_outputs
    archive_test_outputs([folder / name for name in ('result.txt', 'result.running.txt',
                         'config.json', 'summary.json', 'identity.json', 'process_result.json')])
    prepare_test_acceleration(config)
    reference.write_json(folder / 'config.json', config)
    context = nullcontext()
    if method == 'ICS':
        import test_ICS as ics
        ics.install_pics_mode()
    elif method == 'GCSR':
        import test_GCSR as gcsr
        context = gcsr.gosr_environment('one_hop')
    elif method in ('ISACR', 'D3QN+'):
        context = destination_selection(ablation=False)
    with context:
        run_training(config)
    if reference.last_step(running) != reference.EXPECTED_STEP:
        raise RuntimeError(f'Incomplete simulation: {running}')
    running.replace(folder / 'result.txt')
    summarize(constellation, method, seed, folder)
    config['environment']['save_training_data'] = str(folder / 'result.txt')
    reference.write_json(folder / 'config.json', config)
    reference.write_json(folder / 'identity.json', expected)
    reference.write_json(folder / 'process_result.json', dict(exit_code=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, default=15, choices=range(1, 21))
    parser.add_argument('--acceleration', choices=('on', 'off'), default='on')
    parser.add_argument('--case', nargs=3, metavar=('CONSTELLATION', 'METHOD', 'SEED'))
    args = parser.parse_args()
    if args.case:
        constellation, method, seed = args.case
        if constellation not in CONSTELLATIONS or method not in METHODS:
            parser.error('Unknown constellation/method')
        execute(constellation, method, None if seed == 'single' else int(seed), args.acceleration)
        return
    policy = copy.deepcopy(REVIEWED_VALIDATION).get('seven_schedule_policy', {})
    if policy:
        args.workers = policy['workers']
        if policy.get('wait_for_ablation') and not args.dry_run:
            snapshot = reference.read_json(ROOT / 'training_process_data/v1/test/test/runtime/ablation_plus_adaptive_latest.json')
            if snapshot.get('active') or snapshot.get('failed') or len(snapshot.get('completed', [])) != 15:
                raise RuntimeError('Seven-method tests must wait for all fifteen new ablation results')
        print(f'Authorized seven-method worker limit: {args.workers}', flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    def live_worker(worker):
        try:
            return abs(psutil.Process(worker['pid']).create_time() - worker['create_time']) < .01
        except psutil.NoSuchProcess:
            return False
    lock = OUTPUT / 'TESTS_RUNNING.lock'
    validation = copy.deepcopy(REVIEWED_VALIDATION)
    adopted = {tuple(worker['case']): worker for worker in validation.get('seven_orphan_workers', [])}
    if not args.dry_run:
        if lock.exists() and psutil.pid_exists(int(lock.read_text())):
            owner = int(lock.read_text())
            validation = copy.deepcopy(REVIEWED_VALIDATION)
            shared = validation.get('shared_seven_methods', {})
            if shared.get('pid') != owner or not live_worker(shared):
                raise RuntimeError('This test batch is already running')
            print('Waiting for the authorized shared-slot batch to finish.', flush=True)
            while live_worker(shared):
                time.sleep(2)
            previous = reference.read_json(OUTPUT / 'TEST_PLAN.json')
            if any(row.get('status') in ('failed', 'blocked') for row in previous):
                raise RuntimeError('Shared-slot batch needs diagnosed repair before retry')
        lock.unlink(missing_ok=True)
        with lock.open('x') as stream:
            stream.write(str(os.getpid()))
    try:
        plan, jobs = [], []
        for constellation in CONSTELLATIONS:
            for method in METHODS:
                for seed in ((None,) if method in ('ICS', 'GCSR') else range(5)):
                    item = dict(constellation=constellation, method=method, training_seed=seed)
                    try:
                        folder = directory(constellation, method, seed)
                        config = build_config(constellation, method, seed, folder / 'result.txt')
                        if complete(folder, identity(config)) or verified_copy(folder, config):
                            item['status'] = 'complete'
                        elif [constellation, method, seed] in policy.get('excluded_cases', []):
                            item.update(status='failed', reason='User interrupted this case; automatic retry excluded')
                        elif method == 'ISACR':
                            try:
                                import_isacr(constellation, seed, config, dry_run=args.dry_run)
                                item['status'] = 'reuse' if args.dry_run else 'reused'
                            except (OSError, ValueError, KeyError) as error:
                                item.update(status='waiting_for_coefficient_2', reason=str(error))
                        else:
                            item['status'] = 'ready'
                            jobs.append((constellation, method, seed))
                    except (OSError, ValueError, RuntimeError, KeyError) as error:
                        item.update(status='blocked', reason=str(error))
                    plan.append(item)
        reference.write_json(OUTPUT / 'TEST_PLAN.json', plan)
        from collections import Counter
        print(dict(Counter(item['status'] for item in plan)), flush=True)
        if args.dry_run:
            return
        jobs.sort(key=lambda job: job not in adopted)
        slots = threading.Condition()
        running = 0
        def launch(job):
            nonlocal running
            with slots:
                while True:
                    try:
                        snapshot = reference.read_json(ROOT / 'training_process_data/v1/test/test/runtime/ablation_plus_adaptive_latest.json')
                    except PermissionError:
                        slots.wait(timeout=2)
                        continue
                    ablations = (sum(live_worker(worker) for worker in snapshot.get('active', []))
                                 if policy.get('wait_for_ablation', True) else 0)
                    if running + ablations < args.workers:
                        running += 1
                        break
                    slots.wait(timeout=2)
            try:
                return launch_reserved(job)
            finally:
                with slots:
                    running -= 1
                    slots.notify_all()
        def launch_reserved(job):
            constellation, method, seed = job
            folder = directory(*job)
            if job in adopted:
                worker = adopted[job]
                while live_worker(worker):
                    time.sleep(2)
                saved = reference.read_json(folder / 'identity.json')
                if saved != worker['expected_identity']:
                    raise ValueError(f'Adopted case identity differs: {folder}')
                if reference.read_json(folder / 'process_result.json')['exit_code'] != 0:
                    raise ValueError(f'Adopted case failed: {folder}')
                summarize(*job, folder, write=False)
                with slots:
                    evidence = reference.read_json(policy_path)
                    evidence.setdefault('seven_completed_compatibility', {})[str(folder.relative_to(ROOT))] = dict(
                        identity_sha256=reference.digest(folder / 'identity.json'),
                        raw_sha256=reference.digest(folder / 'result.txt'),
                        runtime_ast_sha256=worker['runtime_ast_sha256'],
                        review='Preserved exact live case across scheduler-only repair; exit0 and fixed trace verified')
                    reference.write_json(policy_path, evidence)
                if not complete(folder, identity(build_config(*job, folder / 'result.txt'))):
                    raise ValueError(f'Adopted case acceptance failed: {folder}')
                return job, 0
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'console.txt').open('w', encoding='utf-8') as stream:
                result = subprocess.run([sys.executable, '-B', '-u', '-X', 'faulthandler',
                    str(Path(__file__).resolve()), '--case', constellation, method,
                    'single' if seed is None else str(seed), '--acceleration', args.acceleration],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                reference.write_json(folder / 'process_result.json', dict(exit_code=result.returncode))
            return job, result.returncode
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(launch, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                reason = None
                try:
                    _, code = future.result()
                except (OSError, ValueError, RuntimeError) as error:
                    code, reason = 1, str(error)
                for item in plan:
                    if (item['constellation'], item['method'], item['training_seed']) == job:
                        item['status'] = 'complete' if code == 0 else 'failed'
                        item['exit_code'] = code
                        if reason:
                            item['reason'] = reason
                reference.write_json(OUTPUT / 'TEST_PLAN.json', plan)
                print(job, 'complete' if code == 0 else f'FAILED ({code})', flush=True)
        rows = [reference.read_json(p) for p in OUTPUT.glob('*/*/*/summary.json')
                if p.parent.parent.name in METHODS
                and reference.read_json(p.parent / 'process_result.json').get('exit_code') == 0]
        reference.write_rows(OUTPUT / 'results.csv', rows)
        print('Batch finished; missing models/results remain listed in TEST_PLAN.json.', flush=True)
    finally:
        if not args.dry_run:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault('PYTHONHASHSEED', '2025')
    main()
