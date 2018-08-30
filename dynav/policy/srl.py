import torch
import torch.nn as nn
import logging
from dynav.policy.multi_ped_rl import MultiPedRL


class ValueNetwork(nn.Module):
    def __init__(self, self_state_dim, ped_state_dim, mlp1_dims, mlp2_dims):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.mlp1 = nn.Sequential(nn.Linear(self_state_dim + ped_state_dim, mlp1_dims[0]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[0], mlp1_dims[1]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[1], mlp1_dims[2]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[2], mlp2_dims))
        self.mlp2 = nn.Sequential(nn.Linear(mlp2_dims + self.self_state_dim, 1))

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of peds, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        state = torch.reshape(state, (-1, size[2]))
        pooled_state = torch.max(torch.reshape(self.mlp1(state), (size[0], size[1], -1)), 1)[0]
        joint_state = torch.cat([self_state, pooled_state], dim=1)
        value = self.mlp2(joint_state)
        return value


class SRL(MultiPedRL):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('srl', 'mlp1_dims').split(', ')]
        mlp2_dims = config.getint('srl', 'mlp2_dims')
        self.model = ValueNetwork(self.self_state_dim, self.ped_state_dim, mlp1_dims, mlp2_dims)
        self.multiagent_training = config.getboolean('srl', 'multiagent_training')
        logging.info('SRL: {} agent training'.format('single' if not self.multiagent_training else 'multiple'))
