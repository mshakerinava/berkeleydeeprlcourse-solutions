#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
# [Mehran Shakeriava] change begin
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save', action='store_true')
# [Mehran Shakeriava] change end
    args = parser.parse_args()

# [Mehran Shakeriava] change begin
    import random
    random.seed(args.seed, version=2)
    tf.set_random_seed(random.randint(0, 2**32 - 1))
    np.random.seed(random.randint(0, 2**32 - 1))
# [Mehran Shakeriava] change end

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

# [Mehran Shakeriava] change begin
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # with tf.Session():
    with tf.Session(config=config):
# [Mehran Shakeriava] change end
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
# [Mehran Shakeriava] change begin
        env.seed(random.randint(0, 2**32 - 1))
        # max_steps = args.max_timesteps or env.spec.timestep_limit
        max_steps = args.max_timesteps or env.spec.max_episode_steps
# [Mehran Shakeriava] change end

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

# [Mehran Shakeriava] change begin
        if args.save:
            save_dir = 'expert_data_%d' % args.num_rollouts
            os.makedirs(save_dir, exist_ok=True)
            # with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            with open(os.path.join(save_dir, args.envname + '.pkl'), 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(save_dir, args.envname + '-score.txt'), 'w') as f:
                print(np.mean(returns), '±', np.std(returns), file=f)
# [Mehran Shakeriava] change end

if __name__ == '__main__':
    main()
