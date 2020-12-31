import re
import argparse
import matplotlib.pyplot as plt
import numpy as np


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--plot_sr', default=False, action='store_true')
    parser.add_argument('--plot_cr', default=False, action='store_true')
    parser.add_argument('--plot_time', default=False, action='store_true')
    parser.add_argument('--plot_reward', default=True, action='store_true')
    parser.add_argument('--plot_train', default=True, action='store_true')
    parser.add_argument('--plot_val', default=False, action='store_true')
    parser.add_argument('--window_size', type=int, default=200)
    args = parser.parse_args()

    # define the names of the models you want to plot and the longest episodes you want to show
    models = ['LSTM-RL', 'SARL', 'OM-SARL']
    max_episodes = 10000

    ax1 = ax2 = ax3 = ax4 = None
    ax1_legends = []
    ax2_legends = []
    ax3_legends = []
    ax4_legends = []

    for i, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()

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
        train_episode = train_episode[:max_episodes]
        train_sr = train_sr[:max_episodes]
        train_cr = train_cr[:max_episodes]
        train_time = train_time[:max_episodes]
        train_reward = train_reward[:max_episodes]

        # smooth training plot
        train_sr_smooth = running_mean(train_sr, args.window_size)
        train_cr_smooth = running_mean(train_cr, args.window_size)
        train_time_smooth = running_mean(train_time, args.window_size)
        train_reward_smooth = running_mean(train_reward, args.window_size)

        # plot sr
        if args.plot_sr:
            if ax1 is None:
                _, ax1 = plt.subplots()
            if args.plot_train:
                ax1.plot(range(len(train_sr_smooth)), train_sr_smooth)
                ax1_legends.append(models[i])
            if args.plot_val:
                ax1.plot(val_episode, val_sr)
                ax1_legends.append(models[i])

            ax1.legend(ax1_legends)
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Success Rate')
            ax1.set_title('Success rate')

        # plot time
        if args.plot_time:
            if ax2 is None:
                _, ax2 = plt.subplots()
            if args.plot_train:
                ax2.plot(range(len(train_time_smooth)), train_time_smooth)
                ax2_legends.append(models[i])
            if args.plot_val:
                ax2.plot(val_episode, val_time)
                ax2_legends.append(models[i])

            ax2.legend(ax2_legends)
            ax2.set_xlabel('Episodes')
            ax2.set_ylabel('Time(s)')
            ax2.set_title("Robot's Time to Reach Goal")

        # plot cr
        if args.plot_cr:
            if ax3 is None:
                _, ax3 = plt.subplots()
            if args.plot_train:
                ax3.plot(range(len(train_cr_smooth)), train_cr_smooth)
                ax3_legends.append(models[i])
            if args.plot_val:
                ax3.plot(val_episode, val_cr)
                ax3_legends.append(models[i])

            ax3.legend(ax3_legends)
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Collision Rate')
            ax3.set_title('Collision Rate')

        # plot reward
        if args.plot_reward:
            if ax4 is None:
                _, ax4 = plt.subplots()
            if args.plot_train:
                ax4.plot(range(len(train_reward_smooth)), train_reward_smooth)
                ax4_legends.append(models[i])
            if args.plot_val:
                ax4.plot(val_episode, val_reward)
                ax4_legends.append(models[i])

            ax4.legend(ax4_legends)
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Reward')
            ax4.set_title('Cumulative Discounted Reward')

    plt.show()


if __name__ == '__main__':
    main()
