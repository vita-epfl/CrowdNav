import torch
import logging
import argparse
import configparser
import numpy as np
import gym
from dynav.utils.navigator import Navigator
from dynav.utils.explorer import Explorer
from dynav.policy.policy_factory import policy_factory


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/orca_env.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: {}'.format(device))

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    if policy.trainable:
        if args.weights is None:
            parser.error('Trainable policy must be specified with a saved weights')
        policy.get_model().load_state_dict(torch.load(args.weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    navigator = Navigator(env_config, 'navigator')
    navigator.set_policy(policy)
    env.set_navigator(navigator)
    explorer = Explorer(env, navigator, device)

    policy.set_phase(args.phase)
    policy.set_device(device)
    if args.policy in ['cadrl', 'srl']:
        policy.set_env(env)

    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(navigator.get_position())
        while not done:
            action = navigator.act(ob)
            ob, reward, done, info = env.step(action)
            current_pos = np.array(navigator.get_position())
            logging.debug('Speed: {:.2f}'.format(np.linalg.norm(current_pos - last_pos) / navigator.time_step))
            last_pos = current_pos
        env.render('video', args.output_file)
        logging.info('It takes {:.2f} seconds to finish. Final status is {}'.format(env.timer, info))
    else:
        if args.phase == 'val':
            explorer.run_k_episodes(env.val_size, args.phase, print_failure=True)
        elif args.phase == 'test':
            explorer.run_k_episodes(env.test_size, args.phase, print_failure=True)


if __name__ == '__main__':
    main()
