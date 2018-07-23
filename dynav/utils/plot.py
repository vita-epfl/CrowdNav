import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str)
    parser.add_argument('--plot_sr', default=True, action='store_true')
    parser.add_argument('--plot_cr', default=False, action='store_true')
    parser.add_argument('--plot_time', default=True, action='store_true')
    parser.add_argument('--plot_train', default=True, action='store_true')
    parser.add_argument('--plot_val', default=True, action='store_true')
    parser.add_argument('--plot_test', default=False, action='store_true')
    parser.add_argument('--window_size', type=int, default=10)
    args = parser.parse_args()

    with open(args.log_file, 'r') as fo:
        log = fo.read()

    val_pattern = r"VAL   in episode (?P<episode>\d+) has success rate: (?P<sr>0.\d+), " \
                  r"collision rate: (?P<cr>0.\d+), average time to reach goal: (?P<time>\d+)"
    val_episode = []
    val_sr = []
    val_cr = []
    val_time = []

    for r in re.findall(val_pattern, log):
        val_episode.append(int(r[0]))
        val_sr.append(float(r[1]))
        val_cr.append(float(r[2]))
        val_time.append(int(r[3]))

    test_pattern = r"TEST  in episode (?P<episode>\d+) has success rate: (?P<sr>0.\d+), " \
                   r"collision rate: (?P<cr>0.\d+), average time to reach goal: (?P<time>\d+)"
    test_episode = []
    test_sr = []
    test_cr = []
    test_time = []

    for r in re.findall(test_pattern, log):
        test_episode.append(int(r[0]))
        test_sr.append(float(r[1]))
        test_cr.append(float(r[2]))
        test_time.append(int(r[3]))

    train_pattern = r"TRAIN in episode (?P<episode>\d+) has success rate: (?P<sr>0.\d+), " \
                    r"collision rate: (?P<cr>0.\d+), average time to reach goal: (?P<time>\d+)"
    train_episode = []
    train_sr = []
    train_cr = []
    train_time = []

    for r in re.findall(train_pattern, log):
        train_episode.append(int(r[0]))
        train_sr.append(float(r[1]))
        train_cr.append(float(r[2]))
        train_time.append(int(r[3]))

    # smoothing
    train_episode = np.array(train_episode)
    train_episode_new = np.linspace(train_episode.min(), train_episode.max(), int(len(train_episode)/args.window_size))
    train_sr_smooth = spline(train_episode, train_sr, train_episode_new)
    train_cr_smooth = spline(train_episode, train_cr, train_episode_new)
    train_time_smooth = spline(train_episode, train_time, train_episode_new)

    # plot sr
    if args.plot_sr:
        fig1, ax1 = plt.subplots()
        legends = []
        if args.plot_train:
            ax1.plot(train_episode_new, train_sr_smooth)
            legends.append('train')
        if args.plot_val:
            ax1.plot(val_episode, val_sr)
            legends.append('val')
        if args.plot_test:
            ax1.plot(test_episode, test_sr)
            legends.append('test')
        ax1.legend(legends)
        ax1.set_title('Success rate')

    # plot time
    if args.plot_time:
        fig2, ax2 = plt.subplots()
        legends = []
        if args.plot_train:
            ax2.plot(train_episode_new, train_time_smooth)
            legends.append('train')
        if args.plot_val:
            ax2.plot(val_episode, val_time)
            legends.append('val')
        if args.plot_test:
            ax2.plot(test_episode, test_time)
            legends.append('test')
        ax2.legend(legends)
        ax2.set_title('Time to reach goal')

    # plot cr
    if args.plot_cr:
        fig3, ax3 = plt.subplots()
        legends = []
        if args.plot_train:
            ax3.plot(train_episode_new, train_cr_smooth)
            legends.append('train')
        if args.plot_val:
            ax3.plot(val_episode, val_cr)
            legends.append('val')
        if args.plot_test:
            ax3.plot(test_episode, test_cr)
            legends.append('test')
        ax3.legend(legends)
        ax3.set_title('Collision rate')

    plt.show()


if __name__ == '__main__':
    main()