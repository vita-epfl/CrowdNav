import torch
import logging
import argparse
import configparser
import gym
from dynav.utils.navigator import Navigator
from gym_crowd.envs.policy.policy_factory import policy_factory


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    args = parser.parse_args()

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)

    # configure policy
    policy = policy_factory[args.policy]()
    if policy.trainable:
        if args.weights is None or args.policy_config is None:
            parser.error('Weights file and model config has to be specified for a trainable network')
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)
        policy.configure(policy_config)

    # configure device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: {}'.format(device))

    # configure environment
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    navigator = Navigator(env_config, 'navigator')
    navigator.policy = policy
    env.set_navigator(navigator)

    if args.visualize:
        ob = env.reset('test')
        # env.render()
        timer = 0
        done = False
        while not done:
            action = navigator.act(ob)
            ob, reward, done, info = env.step(action)
            timer += 1
            # env.render()
        env.render('video')
        print('It takes {} steps to finish. Last step is {}'.format(timer, info))


if __name__ == '__main__':
    main()
