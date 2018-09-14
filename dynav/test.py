import torch
import logging
import argparse
import configparser
import numpy as np
import gym
import os
from dynav.utils.navigator import Navigator
from dynav.utils.explorer import Explorer
from dynav.policy.policy_factory import policy_factory
from gym_crowd.envs.policy.orca import ORCA


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/orca_env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: {}'.format(device))

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    navigator = Navigator(env_config, 'navigator')
    navigator.set_policy(policy)
    env.set_navigator(navigator)
    explorer = Explorer(env, navigator, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(navigator.policy, ORCA):
        if navigator.visible:
            navigator.policy.safety_space = 0
        else:
            navigator.policy.safety_space = 0
        logging.info('ORCA agent buffer: {}'.format(navigator.policy.safety_space))

    policy.set_env(env)
    navigator.print_info()
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
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes {:.2f} seconds to finish. Final status is {}'.format(env.global_time, info))
        if navigator.visible and info == 'reach goal':
            ped_times = env.get_ped_times()
            logging.info('Average time for peds to reach goal: {:.2f}'.format(sum(ped_times) / len(ped_times)))
    else:
        nav_times, ped_times, rewards = explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
        if args.model_dir is not None:
            with open(os.path.join(args.model_dir, 'results.txt'), mode='w') as fo:
                fo.write(' '.join([str(time) for time in nav_times]))
                if navigator.visible:
                    fo.write('\n' + ' '.join([str(time) for time in ped_times]))
                fo.write('\n' + ' '.join([str(reward) for reward in rewards]))


if __name__ == '__main__':
    main()
