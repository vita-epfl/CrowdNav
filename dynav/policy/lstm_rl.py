import torch
import torch.nn as nn
import numpy as np
import logging
from dynav.policy.multi_ped_rl import MultiPedRL


class ValueNetwork(nn.Module):
    def __init__(self, self_state_dim, mlp_dims, ped_state_dim, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp = nn.Sequential(nn.Linear(self_state_dim + lstm_hidden_dim, mlp_dims[0]), nn.ReLU(),
                                 nn.Linear(mlp_dims[0], mlp_dims[1]), nn.ReLU(),
                                 nn.Linear(mlp_dims[1], 1))
        self.lstm = nn.LSTM(ped_state_dim, lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of peds, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        ped_state = state[:, :, self.self_state_dim:]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(ped_state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class LstmRL(MultiPedRL):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('lstm_rl', 'mlp_dims').split(', ')]
        lstm_hidden_dim = config.getint('lstm_rl', 'lstm_hidden_dim')
        self_state_dim = 6
        ped_state_dim = self.joint_state_dim - self_state_dim
        self.model = ValueNetwork(self_state_dim, mlp_dims, ped_state_dim, lstm_hidden_dim)
        self.multiagent_training = config.getboolean('lstm_rl', 'multiagent_training')
        logging.info('LSTM-RL: {} agent training'.format('single' if not self.multiagent_training else 'multiple'))

    def predict(self, state):
        """
        Input state is the joint state of navigator concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        def dist(ped):
            # sort ped order by decreasing distance to the navigator
            return np.linalg.norm(np.array(ped.position) - np.array(state.self_state.position))

        state.ped_states = sorted(state.ped_states, key=dist, reverse=True)
        return super().predict(state)

