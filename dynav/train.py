import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import copy
import sys
import logging
import random
import itertools
import argparse
import configparser
import math
import os
import numpy as np
import re
import shutil


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='linear')
    parser.add_argument('--gpu', default=False, action='store_true')
    args = parser.parse_args()
    config_file = args.config
    model_config = configparser.RawConfigParser()
    model_config.read(config_file)
    env_config = configparser.RawConfigParser()
    env_config.read('configs/env.config')

    # configure paths
    output_dir = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = os.path.join('data', output_dir)
    if os.path.exists(output_dir):
        # raise FileExistsError('Output folder already exists')
        print('Output folder already exists')
    else:
        os.mkdir(output_dir)
    log_file = os.path.join(output_dir, 'output.log')
    shutil.copy(args.config, output_dir)
    initialized_weights = os.path.join(output_dir, 'initialized_model.pth')
    trained_weights = os.path.join(output_dir, 'trained_model.pth')

    # configure logging
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # configure device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: {}'.format(device))

    # configure model
    state_dim = model_config.getint('model', 'state_dim')
    kinematic = env_config.getboolean('agent', 'kinematic')
    model = ValueNetwork(state_dim=state_dim, fc_layers=[150, 100, 100], kinematic=kinematic).to(device)
    logging.debug('Trainable parameters: {}'.format([name for name, p in model.named_parameters() if p.requires_grad]))

    # load simulated data from ORCA
    traj_dir = model_config.get('init', 'traj_dir')
    gamma = model_config.getfloat('model', 'gamma')
    capacity = model_config.getint('train', 'capacity')
    memory = initialize_memory(traj_dir, gamma, capacity, kinematic, device)

    # initialize model
    if os.path.exists(initialized_weights):
        model.load_state_dict(torch.load(initialized_weights))
        logging.info('Load initialized model weights')
    else:
        initialize_model(model, memory, model_config, device)
        torch.save(model.state_dict(), initialized_weights)
        logging.info('Finish initializing model. Model saved')

    # train the model
    train(model, memory, model_config, env_config, device, trained_weights)
    torch.save(model.state_dict(), trained_weights)
    logging.info('Finish initializing model. Model saved')


if __name__ == '__main__':
    main()