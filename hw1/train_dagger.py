import os
import json
import pickle
import codecs
import random
import shutil
import hashlib
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from nets import *
import load_policy

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Humanoid-v2')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use for training (case-sensitive)')
parser.add_argument('--nonlinearity', type=str, default='elu', help='Type of nonlinearity used in neural network (tanh, relu, etc.)')
parser.add_argument('--net-size', type=float, default=4.0, help='Multiplier for the size of hidden layers')
parser.add_argument('--grad-clip', type=float, default=0.5, help='Maximum gradient norm')
parser.add_argument('--dropout-prob', type=float, default=0, help='Probability of dropping nodes in Dropout')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='L2-regularizer strength')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate used in optimization')
parser.add_argument('--aggregation-rate', type=float, default=0.2, help='Rate of dataset aggregation')
parser.add_argument('--seed', type=int, default=1, help='Manual PRNG seed for reproducibility')
parser.add_argument('--only-mean', type=int, default=1, help='Only optimize mean of distribution for this many epochs (-1 means always)')
parser.add_argument('--num-demos', type=int, default=20, help='Number of expert demonstrations to learn from')
parser.add_argument('--num-layers', type=int, default=3, help='Number of layers in neural network')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size used for training and evaluation')
parser.add_argument('--num-epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--cpu', action='store_false', dest='gpu')
parser.add_argument('--render', action='store_true')
parser.add_argument('--record', action='store_true')
ARGS = parser.parse_args()

ARGS_STR = ''
for key, val in sorted(vars(ARGS).items()):
    if key not in ['seed', 'gpu', 'render', 'record', 'env']:
        ARGS_STR += str(val)
ARGS_HASH = hashlib.md5(str.encode(ARGS_STR)).hexdigest()[-8:]

SAVE_PATH = './output_dagger/%s/%s/seed_%d' % (ARGS_HASH, ARGS.env, ARGS.seed)
shutil.rmtree(SAVE_PATH, ignore_errors=True)
os.makedirs(SAVE_PATH, exist_ok=True)


def log(s, disp=True, write=False, filename='log.txt', **kwargs):
    if disp:
        print(s, **kwargs)
    if write:
        with codecs.open(os.path.join(SAVE_PATH, filename), 'a', 'utf-8') as f:
            print(s, file=f, **kwargs)


def log_tabular(vals, keys=None, formats=None):
    log(','.join([str(x) for x in vals]), disp=False, write=True, filename='log.csv')
    if formats is not None:
        assert len(formats) == len(vals)
        vals = [x[0] % x[1] for x in zip(formats, vals)]
    if keys is not None:
        assert len(keys) == len(vals)
        log(' | '.join(['%s: %s' % (x[0], str(x[1])) for x in zip(keys, vals)]), write=True)


ARGS.nonlinearity = ARGS.nonlinearity.lower()
ARGS_JSON = json.dumps(vars(ARGS), sort_keys=True, indent=4)
log('ARGS = %s' % ARGS_JSON)
ARGS.optimizer = eval('optim.%s' % ARGS.optimizer)
ARGS.nonlinearity = eval('F.%s' % ARGS.nonlinearity)

with open(os.path.join(SAVE_PATH, 'args.json'), 'w') as f:
    f.write(ARGS_JSON)

# reproducibility
random.seed(ARGS.seed, version=2)
torch.manual_seed(random.randint(0, 2**32 - 1))
np.random.seed(random.randint(0, 2**32 - 1))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEMO_PATH = 'expert_data_%d/%s.pkl' % (ARGS.num_demos, ARGS.env)
with open(DEMO_PATH, 'rb') as f:
    demo = pickle.load(f)

DEMO_SIZE = demo['observations'].shape[0]
assert demo['actions'].shape[0] == DEMO_SIZE

demo['observations'] = np.reshape(demo['observations'], (DEMO_SIZE, -1))
demo['actions'] = np.reshape(demo['actions'], (DEMO_SIZE, -1))

OBS_DIM = demo['observations'].shape[1]
ACT_DIM = demo['actions'].shape[1]

# observation stats used for normalization
obs_mean = np.mean(demo['observations'], axis=0)
obs_mean_sqr = np.mean(demo['observations'] ** 2, axis=0)
obs_std = lambda m1, m2: np.maximum(m2 - m1 ** 2, 0) ** 0.5
count = DEMO_SIZE

log('Number of time-steps in demonstrations: %d' % DEMO_SIZE)
log('Dimensionality of observation-space: %d' % OBS_DIM)
log('Dimensionality of action-space: %d' % ACT_DIM)

# 90% / 10% split of dataset into training-set / evaluation-set
EVAL_SIZE = DEMO_SIZE // 10
TRAIN_SIZE = DEMO_SIZE - EVAL_SIZE

shuffle_idx = np.arange(DEMO_SIZE)
np.random.shuffle(shuffle_idx)

demo_eval = {
    'observations': demo['observations'][shuffle_idx[:EVAL_SIZE]],
    'actions': demo['actions'][shuffle_idx[:EVAL_SIZE]]
}

demo_train = {
    'observations': demo['observations'][shuffle_idx[EVAL_SIZE:]],
    'actions': demo['actions'][shuffle_idx[EVAL_SIZE:]]
}

DEVICE = torch.device('cuda' if ARGS.gpu and torch.cuda.is_available() else 'cpu')
log('DEVICE = %s' % str(DEVICE))

WIDTH = int(np.sqrt(OBS_DIM * ACT_DIM) * ARGS.net_size + 16)
log('Width of hidden layers: %d' % WIDTH, write=True)
hidden_layers = [WIDTH] * (ARGS.num_layers - 1)
net = ControlNet(widths=[OBS_DIM] + hidden_layers + [ACT_DIM], act_fn=ARGS.nonlinearity, dropout_prob=ARGS.dropout_prob)
net.set_obs_stats(obs_mean=obs_mean, obs_std=obs_std(obs_mean, obs_mean_sqr))
print(net)
net.to(DEVICE)
opt = ARGS.optimizer(net.parameters(), lr=ARGS.learning_rate, weight_decay=ARGS.weight_decay)

expert_policy = load_policy.load_policy('./experts/%s.pkl' % ARGS.env)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

env = gym.make(ARGS.env)
env.seed(ARGS.seed)
if ARGS.record:
    env = gym.wrappers.Monitor(env, directory=SAVE_PATH, video_callable=lambda x: True)


def update_stats(observations, obs_mean, obs_mean_sqr, count, net):
    n = observations.shape[0]
    obs_mean = (obs_mean * count + n * np.mean(observations, axis=0)) / (count + n)
    obs_mean_sqr = (obs_mean_sqr * count + n * np.mean(observations ** 2, axis=0)) / (count + n)
    count += n
    net.set_obs_stats(obs_mean=obs_mean, obs_std=obs_std(obs_mean, obs_mean_sqr))
    return obs_mean, obs_mean_sqr, count


def aggregate_dataset(demo_train, demo_eval, observations, expert_policy, sess):
    with sess.as_default():
        actions = expert_policy(observations)
    eval_size = observations.shape[0] // 10
    new_train = {
        'observations': observations[eval_size:],
        'actions': actions[eval_size:]
    }
    new_eval = {
        'observations': observations[:eval_size],
        'actions': actions[:eval_size]
    }
    for key in demo_train:
        demo_train[key] = np.concatenate((demo_train[key], new_train[key]), axis=0)
    for key in demo_eval:
        demo_eval[key] = np.concatenate((demo_eval[key], new_eval[key]), axis=0)
    return demo_train, demo_eval


def train_batch(net, opt, obs_batch, act_batch, only_mean=False):
    net.train()
    mu, log_var = net.forward(obs_batch)
    if only_mean:
        log_var.zero_()
    loss = net.loss(mu, log_var, act_batch)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), ARGS.grad_clip)
    opt.step()
    loss = loss.item()
    return loss


def eval_batch(net, obs_batch, act_batch):
    net.eval()
    with torch.no_grad():
        mu, log_var = net.forward(obs_batch)
        loss = net.loss(mu, log_var, act_batch)
        loss = loss.item()
    return loss


def run_epoch(net, opt, demo, train, num_samples=None, **kwargs):
    dataset_size = demo['observations'].shape[0]
    assert demo['actions'].shape[0] == dataset_size
    if train:
        shuffle_idx = np.arange(dataset_size)
        np.random.shuffle(shuffle_idx)
        demo['observations'] = demo['observations'][shuffle_idx]
        demo['actions'] = demo['actions'][shuffle_idx]
    i = 0
    total_loss = 0
    if num_samples is None:
        num_samples = dataset_size
    while i < num_samples:
        obs_batch = demo['observations'][i: min(i + ARGS.batch_size, num_samples)]
        act_batch = demo['actions'][i: min(i + ARGS.batch_size, num_samples)]
        cur_batch_size = obs_batch.shape[0]
        i += ARGS.batch_size
        obs_batch = torch.tensor(obs_batch, dtype=torch.float, device=DEVICE)
        act_batch = torch.tensor(act_batch, dtype=torch.float, device=DEVICE)
        if train:
            loss = train_batch(net, opt, obs_batch, act_batch, **kwargs)
        else:
            loss = eval_batch(net, obs_batch, act_batch, **kwargs)
        total_loss += cur_batch_size * loss
    total_loss /= num_samples
    return total_loss


def run_policy(net, env, render=False):
    net.eval()
    done = False
    total_reward = 0
    obs_list = []
    obs = env.reset()
    if render:
        env.render()
    while not done:
        obs_list.append(obs)
        with torch.no_grad():
            obs = torch.tensor([obs], dtype=torch.float, device=DEVICE)
            mu, log_var = net.forward(obs)
            mu = mu.cpu().numpy()[0]
            log_var = log_var.cpu().numpy()[0]
        sigma = np.exp(log_var * 0.5)
        act = np.random.normal(loc=mu, scale=sigma)
        if np.isnan(act).any():
            log('WARNING: `act` contains NaN.')
            act = np.where(np.isnan(act), 0, act)
        if (np.abs(act) > 1e10).any():
            log('WARNING: `act` contains a number that is too large.')
        act = np.maximum(env.action_space.low, np.minimum(env.action_space.high, act))
        obs, rew, done, _ = env.step(act)
        if render:
            env.render()
        total_reward += rew
    observations = np.stack(obs_list, axis=0)
    return total_reward, observations


def eval_policy(net, env, runs=10, render=False):
    score_list = []
    observations_list = []
    for k in range(runs):
        score, observations = run_policy(net, env, render=ARGS.render)
        score_list.append(score)
        observations_list.append(observations)
    score_mean = np.mean(score_list)
    score_std = np.std(score_list)
    return score_mean, score_std, observations_list


def save_net(net, cur_score_mean, best_score_mean):
    torch.save(net.state_dict(), os.path.join(SAVE_PATH, 'net_latest.tar'))
    if cur_score_mean > best_score_mean:
        torch.save(net.state_dict(), os.path.join(SAVE_PATH, 'net_best.tar'))
        best_score_mean = cur_score_mean
    return best_score_mean


keys = ['Epoch', 'Score Mean', 'Score Std', 'Training Loss', 'Validation Loss']
formats = ['%03d', '%9.3f', '%9.3f', '%7.4f', '%7.4f']

score_mean, score_std, _ = eval_policy(net, env, render=ARGS.render)
log_tabular(vals=keys)
log_tabular(vals=[0, score_mean, score_std, float('nan'), float('nan')], keys=keys, formats=formats)

best_score_mean = -np.inf
best_score_mean = save_net(net, score_mean, best_score_mean)

for epoch_t in range(ARGS.num_epochs):
    train_loss = run_epoch(net, opt, demo_train, train=True, num_samples=TRAIN_SIZE,
        only_mean=True if ARGS.only_mean == -1 or epoch_t < ARGS.only_mean else False)
    eval_loss = run_epoch(net, opt, demo_eval, train=False)
    score_mean, score_std, observations_list = eval_policy(net, env, render=ARGS.render)
    best_score_mean = save_net(net, score_mean, best_score_mean)
    log_tabular(vals=[epoch_t + 1, score_mean, score_std, train_loss, eval_loss], keys=keys, formats=formats)
    if (epoch_t + 1) % int(1 / ARGS.aggregation_rate) == 0:
        aggregate_dataset(demo_train, demo_eval, observations_list[0], expert_policy, sess)
        if 'Humanoid' in ARGS.env and epoch_t < 5 * int(1 / ARGS.aggregation_rate):
            obs_mean, obs_mean_sqr, count = update_stats(observations_list[0], obs_mean, obs_mean_sqr, count, net)

log('Best Mean Score: %8.3f' % best_score_mean, write=True)
sess.close()
