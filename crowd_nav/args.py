import argparse

class Parser:
    def __init__(self,mode='train'):
        self.parser = argparse.ArgumentParser(description="Arguments for training, test, or plotting")
        available_modes = ['train', 'test', 'plot']

        if mode == 'train' or mode == 'test':
            self.parser.add_argument('--env_config', type=str, default='configs/env.config')
            self.parser.add_argument('--policy_config', type=str, default='configs/policy.config')
            self.parser.add_argument('--train_config', type=str, default='configs/train.config') # to visualize curriculum learning
            self.parser.add_argument('--policy', type=str, default='cadrl')
            self.parser.add_argument('--gpu', default=False, action='store_true')
            self.parser.add_argument('--debug', default=False, action='store_true')
            if mode == 'train':
                self.parser.add_argument('--output_dir', type=str, default='data/output')
                self.parser.add_argument('--weights', type=str)
                self.parser.add_argument('--resume', default=False, action='store_true')
            elif mode == 'test':
                self.parser.add_argument('--model_dir', type=str, default=None)
                self.parser.add_argument('--il', default=False, action='store_true')
                self.parser.add_argument('--visualize', default=False, action='store_true')
                self.parser.add_argument('--phase', type=str, default='test')
                self.parser.add_argument('--test_case', type=int, default=None)
                self.parser.add_argument('--square', default=False, action='store_true')
                self.parser.add_argument('--circle', default=False, action='store_true')
                self.parser.add_argument('--video_file', type=str, default=None)
                self.parser.add_argument('--traj', default=False, action='store_true')
        elif mode == 'plot':
            self.parser.add_argument('log_files', type=str, nargs='+')
            self.parser.add_argument('--plot_sr', default=False, action='store_true')
            self.parser.add_argument('--plot_cr', default=False, action='store_true')
            self.parser.add_argument('--plot_time', default=False, action='store_true')
            self.parser.add_argument('--plot_reward', default=True, action='store_true')
            self.parser.add_argument('--plot_train', default=True, action='store_true')
            self.parser.add_argument('--plot_val', default=False, action='store_true')
            self.parser.add_argument('--plot_epsilon', default=False, action='store_true')
            self.parser.add_argument('--window_size', type=int, default=200)

    def parse(self):
        self.call_args = self.parser.parse_args()
        return self.call_args
