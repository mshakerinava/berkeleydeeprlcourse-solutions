# [Mehran Shakerinava] change begin
import time
import json
# [Mehran Shakerinava] change end
import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            # [Mehran Shakerinava] change begin
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, padding='SAME',
                activation_fn=tf.nn.relu, weights_initializer=tf.orthogonal_initializer())
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, padding='SAME',
                activation_fn=tf.nn.relu, weights_initializer=tf.orthogonal_initializer())
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, padding='SAME',
                activation_fn=tf.nn.relu, weights_initializer=tf.orthogonal_initializer())
            # [Mehran Shakerinava] change end
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

# [Mehran Shakerinava] change begin
def atari_learn(env, session, discount, num_timesteps, batch_size, double, target_update_freq, **kwargs):
# [Mehran Shakerinava] change end
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env=env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        grad_norm_clipping=10,
# [Mehran Shakerinava] change begin
        target_update_freq=target_update_freq,
        batch_size=batch_size,
        gamma=discount,
        double_q=double
# [Mehran Shakerinava] change end
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

# [Mehran Shakerinava] change begin
def set_global_seeds(seed):
    random.seed(seed)
    tf.set_random_seed(random.randint(0, 2 ** 31 - 1))
    np.random.seed(random.randint(0, 2 ** 31 - 1))
# [Mehran Shakerinava] change end

def get_session():
    tf.reset_default_graph()
# [Mehran Shakerinava] change begin
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
# [Mehran Shakerinava] change end
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

# [Mehran Shakerinava] change begin
def get_env(env_name, seed):
    env = gym.make(env_name)

    set_global_seeds(seed)
    env.seed(random.randint(0, 2 ** 31 - 1))

    expt_dir = './vid_dir/' + time.strftime("%d-%m-%Y_%H-%M-%S") + '_' + env_name
    env = wrappers.Monitor(env, expt_dir)
    env = wrap_deepmind(env)

    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-timesteps', type=int, default=2e7)
    parser.add_argument('--target-update-freq', type=int, default=1e4)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--double', action='store_true')
    ARGS = parser.parse_args()
    ARGS_JSON = json.dumps(vars(ARGS), sort_keys=True, indent=4)
    print('ARGS = %s' % ARGS_JSON)

    # Run training
    env = get_env(ARGS.env_id, ARGS.seed)
    session = get_session()
    atari_learn(env, session, **vars(ARGS))
# [Mehran Shakerinava] change end

if __name__ == "__main__":
    main()
