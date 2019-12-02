import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs='*')
parser.add_argument('--labels', type=str, nargs='*')
parser.add_argument('--title', type=str, required=True)
args = parser.parse_args()

data = []
for fname in args.files:
    x = []
    y = []
    with open(fname) as f:
        for line in f.readlines():
            line = line.strip()
            if 'Timestep' in line:
                t = float(line.split()[-1])
                x.append(t)
            if 'mean reward (100 episodes)' in line:
                r = float(line.split()[-1])
                y.append(r)
    data.append((x, y))

with plt.style.context('seaborn'):
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(4, 4))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title(args.title)
    for i in range(len(data)):
        x, y = data[i]
        plt.plot(x, y)#, label=args.labels[i])
    plt.xlabel('Timestep')
    plt.ylabel('Mean Reward')
    # plt.legend(loc='best', fontsize=10)
    # fig.savefig('%s.pdf' % args.title, bbox_inches='tight')
    # fig.savefig('%s.png' % args.title, bbox_inches='tight', dpi=300)
    fig.savefig('%s.svg' % args.title, bbox_inches='tight')
    plt.close()
