import os
import glob
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--hash-bc', type=str, required=True)
parser.add_argument('--hash-dagger', type=str, required=True)
parser.add_argument('-x', type=str)
parser.add_argument('-y', type=str, required=True)
parser.add_argument('--xlabel', type=str)
parser.add_argument('--ylabel', type=str)
parser.add_argument('--title', type=str)
ARGS = parser.parse_args()
ARGS.y = ARGS.y.lower()


def parse_csv(file):
    ret = {}
    with open(file, mode='r') as f:
        lines = [x.strip() for x in f.readlines()]
        keys = [x.strip().lower() for x in lines[0].split(',')]
        for key in keys:
            ret[key] = []
        for line in lines[1:]:
            vals = [x.strip() for x in line.split(',')]
            assert len(vals) == len(keys)
            for i in range(len(vals)):
                ret[keys[i]].append(vals[i])
    return ret, keys


DATA_PATH_DAGGER = './output_dagger/%s/%s' % (ARGS.hash_dagger, ARGS.env)
y_dagger = []
paths = glob.glob(os.path.join(DATA_PATH_DAGGER, 'seed_*'))
for path in paths:
    file = os.path.join(path, 'log.csv')
    d, keys = parse_csv(file)
    y_dagger.append([float(x) for x in d[ARGS.y]])

DATA_PATH_BC = './output_bc/%s/%s' % (ARGS.hash_bc, ARGS.env)
y_bc = []
paths = glob.glob(os.path.join(DATA_PATH_BC, 'seed_*'))
for path in paths:
    file = os.path.join(path, 'log.csv')
    d, keys = parse_csv(file)
    y_bc.append([float(x) for x in d[ARGS.y]])

ARGS.x = ARGS.x if ARGS.x else keys[0]
x = np.array([int(x) for x in d[ARGS.x]])
y_bc = np.array(y_bc)
y_dagger = np.array(y_dagger)

with open(os.path.join(path, 'args.json'), mode='r') as f:
    num_demos = json.load(f)['num_demos']

with open('./expert_data_%d/%s-score.txt' % (num_demos, ARGS.env), mode='r') as f:
    expert_score = float(f.readlines()[0].split('±')[0].strip())

with plt.style.context('seaborn'):
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(4, 4))
    plt.plot(x, y_bc.mean(axis=0), label='BC')
    plt.plot(x, y_dagger.mean(axis=0), label='DAgger')
    plt.plot([x[0], x[-1]], [expert_score, expert_score], label='Expert')
    plt.autoscale(False)
    plt.fill_between(x, y_bc.mean(axis=0) - y_bc.std(axis=0), y_bc.mean(axis=0) + y_bc.std(axis=0), alpha=0.25)
    plt.fill_between(x, y_dagger.mean(axis=0) - y_dagger.std(axis=0), y_dagger.mean(axis=0) + y_dagger.std(axis=0), alpha=0.25)
    plt.title(ARGS.title if ARGS.title else ARGS.env.split('-')[0])
    plt.xlabel(ARGS.xlabel if ARGS.xlabel else ARGS.x.title())
    plt.ylabel(ARGS.ylabel if ARGS.ylabel else ARGS.y.title())
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_PATH_BC, 'plot_x=%s_y=%s.pdf' % (ARGS.x, ARGS.y)), bbox_inches='tight')
    fig.savefig(os.path.join(DATA_PATH_BC, 'plot_x=%s_y=%s.png' % (ARGS.x, ARGS.y)), bbox_inches='tight', dpi=600)
    fig.savefig(os.path.join(DATA_PATH_BC, 'plot_x=%s_y=%s.svg' % (ARGS.x, ARGS.y)), bbox_inches='tight')
    fig.savefig(os.path.join(DATA_PATH_DAGGER, 'plot_x=%s_y=%s.pdf' % (ARGS.x, ARGS.y)), bbox_inches='tight')
    fig.savefig(os.path.join(DATA_PATH_DAGGER, 'plot_x=%s_y=%s.png' % (ARGS.x, ARGS.y)), bbox_inches='tight', dpi=600)
    fig.savefig(os.path.join(DATA_PATH_DAGGER, 'plot_x=%s_y=%s.svg' % (ARGS.x, ARGS.y)), bbox_inches='tight')
    # plt.show()
    plt.close()
