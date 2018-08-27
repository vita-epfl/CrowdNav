import re
import argparse
import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str)
    parser.add_argument('--plot_sr', default=True, action='store_true')
    parser.add_argument('--plot_cr', default=False, action='store_true')
    parser.add_argument('--plot_time', default=True, action='store_true')
    parser.add_argument('--plot_reward', default=True, action='store_true')
    parser.add_argument('--plot_train', default=True, action='store_true')
    parser.add_argument('--plot_val', default=True, action='store_true')
    parser.add_argument('--window_size', type=int, default=100)
    args = parser.parse_args()

    with open(args.log_file, 'r') as fo:
        log = fo.read()

    val_pattern = r"VAL   in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                  r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                  r"total reward: (?P<reward>[-+]?\d+.\d+)"
    val_episode = []
    val_sr = []
    val_cr = []
    val_time = []
    val_reward = []
    for r in re.findall(val_pattern, log):
        val_episode.append(int(r[0]))
        val_sr.append(float(r[1]))
        val_cr.append(float(r[2]))
        val_time.append(float(r[3]))
        val_reward.append(float(r[4]))

    train_pattern = r"TRAIN in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                    r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                    r"total reward: (?P<reward>[-+]?\d+.\d+)"
    train_episode = []
    train_sr = []
    train_cr = []
    train_time = []
    train_reward = []
    for r in re.findall(train_pattern, log):
        train_episode.append(int(r[0]))
        train_sr.append(float(r[1]))
        train_cr.append(float(r[2]))
        train_time.append(float(r[3]))
        train_reward.append(float(r[4]))

    # smooth training plot
    train_sr_smooth = running_mean(train_sr, args.window_size)
    train_cr_smooth = running_mean(train_cr, args.window_size)
    train_time_smooth = running_mean(train_time, args.window_size)
    train_reward_smooth = running_mean(train_reward, args.window_size)

    # plot sr
    if args.plot_sr:
        fig1, ax1 = plt.subplots()
        legends = []
        if args.plot_train:
            ax1.plot(range(len(train_sr_smooth)), train_sr_smooth)
            legends.append('train')
        if args.plot_val:
            ax1.plot(val_episode, val_sr)
            legends.append('val')
        ax1.legend(legends)
        ax1.set_title('Success rate')

    # plot time
    if args.plot_time:
        fig2, ax2 = plt.subplots()
        legends = []
        if args.plot_train:
            ax2.plot(range(len(train_time_smooth)), train_time_smooth)
            legends.append('train')
        if args.plot_val:
            ax2.plot(val_episode, val_time)
            legends.append('val')
        ax2.legend(legends)
        ax2.set_title("Navigator's time to reach goal")

    # plot cr
    if args.plot_cr:
        fig3, ax3 = plt.subplots()
        legends = []
        if args.plot_train:
            ax3.plot(range(len(train_cr_smooth)), train_cr_smooth)
            legends.append('train')
        if args.plot_val:
            ax3.plot(val_episode, val_cr)
            legends.append('val')
        ax3.legend(legends)
        ax3.set_title('Collision rate')

    # plot reward
    if args.plot_reward:
        fig4, ax4 = plt.subplots()
        legends = []
        if args.plot_train:
            ax4.plot(range(len(train_reward_smooth)), train_reward_smooth)
            legends.append('train')
        if args.plot_val:
            ax4.plot(val_episode, val_reward)
            legends.append('val')
        ax4.legend(legends)
        ax4.set_title('Cumulative reward')

    plt.show()


if __name__ == '__main__':
    main()
