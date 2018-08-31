import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from dynav.policy.multi_ped_rl import MultiPedRL


class ValueNetwork(nn.Module):
    def __init__(self, self_state_dim, ped_state_dim, mlp1_dims, mlp2_dims, attention_dims, global_state_dim,
                 input_dim=None):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = global_state_dim
        mlp1_input_dim = input_dim if input_dim is not None else self_state_dim + ped_state_dim
        self.mlp1 = nn.Sequential(nn.Linear(mlp1_input_dim, mlp1_dims[0]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[0], mlp1_dims[1]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[1], global_state_dim))
        self.attention = nn.Sequential(nn.Linear(global_state_dim * 2, attention_dims[0]), nn.ReLU(),
                                       nn.Linear(attention_dims[0], attention_dims[1]), nn.ReLU(),
                                       nn.Linear(attention_dims[1], 1), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(global_state_dim + self.self_state_dim, mlp2_dims[0]), nn.ReLU(),
                                  nn.Linear(mlp2_dims[0], mlp2_dims[1]), nn.ReLU(),
                                  nn.Linear(mlp2_dims[1], mlp2_dims[2]), nn.ReLU(),
                                  nn.Linear(mlp2_dims[1], 1))
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of peds, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        # calculate the global state
        global_state = torch.mean(torch.reshape(mlp1_output, (size[0], size[1], -1)), 1, keepdim=True)
        global_state = torch.reshape(global_state.expand((size[0], size[1], self.global_state_dim)),
                                     (-1, self.global_state_dim))
        # concatenate pairwise interaction with global state to compute attention score
        attention_input = torch.cat([mlp1_output, global_state], dim=1)
        scores = torch.reshape(self.attention(attention_input), (size[0], size[1], 1)).squeeze(dim=2)
        weights = softmax(scores, dim=1).unsqueeze(2)
        # for visualization purpose
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        features = torch.reshape(mlp1_output, (size[0], size[1], -1))
        weighted_feature = torch.sum(weights.expand_as(features) * features, dim=1)
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp2(joint_state)
        return value


class SARL(MultiPedRL):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        global_state_dim = config.getint('sarl', 'global_state_dim')
        self.model = ValueNetwork(self.self_state_dim, self.ped_state_dim, mlp1_dims, mlp2_dims, attention_dims,
                                  global_state_dim)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        logging.info('SARL: {} agent training'.format('single' if not self.multiagent_training else 'multiple'))

    def get_attention_weights(self):
        return self.model.attention_weights
