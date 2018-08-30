import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from dynav.policy.multi_ped_rl import MultiPedRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp1_dims, mlp2_dims):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(input_dim, mlp1_dims[0]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[0], mlp1_dims[1]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[1], mlp1_dims[2]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[2], mlp2_dims))
        self.mlp2 = nn.Sequential(nn.Linear(mlp2_dims, 1))
        self.attention = nn.Sequential(nn.Linear(mlp2_dims, 50), nn.ReLU(),
                                       nn.Linear(50, 1))
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of peds, length of a rotated state)
        :return:
        """
        size = state.shape
        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        scores = torch.reshape(self.attention(mlp1_output), (size[0], size[1], 1)).squeeze(dim=2)
        weights = softmax(scores, dim=1).unsqueeze(2)
        # for visualization purpose
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        features = torch.reshape(mlp1_output, (size[0], size[1], -1))
        weighted_feature = torch.sum(weights.expand_as(features) * features, dim=1)
        value = self.mlp2(weighted_feature)
        return value


class SARL(MultiPedRL):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = config.getint('sarl', 'mlp2_dims')
        self.model = ValueNetwork(self.joint_state_dim, mlp1_dims, mlp2_dims)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        logging.info('SARL: {} agent training'.format('single' if not self.multiagent_training else 'multiple'))

    def get_attention_weights(self):
        return self.model.attention_weights
