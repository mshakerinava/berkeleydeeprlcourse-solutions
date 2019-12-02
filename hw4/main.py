import os
import argparse
import time

from half_cheetah_env import HalfCheetahEnv
from logger import logger
from model_based_rl import ModelBasedRL

parser = argparse.ArgumentParser()
parser.add_argument('question', type=str, choices=('q1, q2, q3'))
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--env', type=str, default='HalfCheetah', choices=('HalfCheetah',))
parser.add_argument('--render', action='store_true')
parser.add_argument('--mpc_horizon', type=int, default=15)
parser.add_argument('--num_random_action_selection', type=int, default=4096)
parser.add_argument('--nn_layers', type=int, default=1)
# [Mehran Shakerinava] change begin
parser.add_argument('--seed', type=int, default=1)
# [Mehran Shakerinava] change end
args = parser.parse_args()

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
exp_name = '{0}_{1}_{2}'.format(args.env,
                                args.question,
                                args.exp_name if args.exp_name else time.strftime("%d-%m-%Y_%H-%M-%S"))
exp_dir = os.path.join(data_dir, exp_name)
# [Mehran Shakerinava] change begin
# assert not os.path.exists(exp_dir),\
#     'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(exp_dir)
import shutil
shutil.rmtree(exp_dir, ignore_errors=True)
# [Mehran Shakerinava] change end
os.makedirs(exp_dir, exist_ok=True)
logger.setup(exp_name, os.path.join(exp_dir, 'log.txt'), 'debug')

env = {
    'HalfCheetah': HalfCheetahEnv()
}[args.env]

# [Mehran Shakeriava] change begin
import random
import numpy as np
import tensorflow as tf
random.seed(args.seed, version=2)
tf.set_random_seed(random.randint(0, 2**32 - 1))
np.random.seed(random.randint(0, 2**32 - 1))
env.seed(random.randint(0, 2**32 - 1))
# [Mehran Shakeriava] change end

mbrl = ModelBasedRL(env=env,
                    render=args.render,
                    mpc_horizon=args.mpc_horizon,
                    num_random_action_selection=args.num_random_action_selection,
                    nn_layers=args.nn_layers)

run_func = {
    'q1': mbrl.run_q1,
    'q2': mbrl.run_q2,
    'q3': mbrl.run_q3
}[args.question]
run_func()
