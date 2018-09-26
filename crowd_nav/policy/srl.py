import torch
import torch.nn as nn
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp2 = mlp(mlp1_dims[-1] + self.self_state_dim, mlp2_dims)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        state = torch.reshape(state, (-1, size[2]))
        pooled_state = torch.max(torch.reshape(self.mlp1(state), (size[0], size[1], -1)), 1)[0]
        joint_state = torch.cat([self_state, pooled_state], dim=1)
        value = self.mlp2(joint_state)
        return value


class SRL(MultiHumanRL):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('srl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('srl', 'mlp2_dims').split(', ')]
        self.with_om = config.getboolean('srl', 'with_om')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims)
        self.multiagent_training = config.getboolean('srl', 'multiagent_training')
        logging.info('Policy: {}SRL'.format('OM-' if self.with_om else ''))
