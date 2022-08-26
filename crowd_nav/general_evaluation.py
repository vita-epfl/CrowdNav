# Script to serialize testing
import logging
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.args import Parser
import sys


def main():
    parser = Parser(mode='test')
    args = parser.parse()
    video_file = None
    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.video_file is not None:
            video_file = os.path.join(args.model_dir, os.path.basename(args.video_file))
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
    log_file = os.path.join(args.model_dir, 'general_evaluation.log')
    file_handler = logging.FileHandler(log_file, mode = 'w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights,map_location=device))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # # set safety space for ORCA in non-cooperative simulation
    # if isinstance(robot.policy, ORCA):
    #     if robot.visible:
    #         robot.policy.safety_space = 0
    #     else:
    #         # because invisible case breaks the reciprocal assumption
    #         # adding some safety space improves ORCA performance. Tune this value based on your need.
    #         robot.policy.safety_space = 0
    #     logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    n_humans = [0, 5, 10, 15]
    n_obstacles = [0, 5, 10, 15]

    n_fixed_humans = 5
    n_fixed_obstacles = 5

    logging.info("General Evaluation: (Fixed #obstacles)")
    ## Start with fixed number of obstacles
    for i, n in enumerate(n_humans):
        env.human_num = n
        env.static_obstacle_num = n_fixed_obstacles
        logging.info("(#humans: {}, #obstacles: {})"
        .format(env.human_num, env.static_obstacle_num)
        )
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    logging.info("General Evaluation: (Fixed #humans)")
    for i, n in enumerate(n_obstacles):
        env.human_num = n_fixed_humans
        env.static_obstacle_num = n
        logging.info("(#humans: {}, #obstacles: {})"
        .format(env.human_num, env.static_obstacle_num)
        )
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    


if __name__ == '__main__':
    main()
