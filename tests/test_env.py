from cmath import phase
import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
import numpy as np

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='random')
    
    args = parser.parse_args()

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    # print(env_config.sections())
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    
    done = False

    ob = env.reset(phase='test', test_case=-2)
    last_pos = np.array(robot.get_position())
    while not done:
        action = robot.act(ob)
        ob, _, done, info = env.step(action)
        current_pos = np.array(robot.get_position())
        logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
        last_pos = current_pos
    env.render('video')

if __name__ == '__main__':
    main()
