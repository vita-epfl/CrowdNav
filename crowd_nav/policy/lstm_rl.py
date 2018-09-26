import torch
import torch.nn as nn
import numpy as np
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork1(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        # human_state = state[:, :, self.self_state_dim:]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(mlp1_dims[-1], lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]

        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        mlp1_output = torch.reshape(mlp1_output, (size[0], size[1], -1))

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(mlp1_output, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class LstmRL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'LSTM-RL'
        self.with_interaction_module = None
        self.interaction_module_dims = None

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('lstm_rl', 'mlp2_dims').split(', ')]
        global_state_dim = config.getint('lstm_rl', 'global_state_dim')
        self.with_om = config.getboolean('lstm_rl', 'with_om')
        with_interaction_module = config.getboolean('lstm_rl', 'with_interaction_module')
        if with_interaction_module:
            mlp1_dims = [int(x) for x in config.get('lstm_rl', 'mlp1_dims').split(', ')]
            self.model = ValueNetwork2(self.input_dim(), self.self_state_dim, mlp1_dims, mlp_dims, global_state_dim)
        else:
            self.model = ValueNetwork1(self.input_dim(), self.self_state_dim, mlp_dims, global_state_dim)
        self.multiagent_training = config.getboolean('lstm_rl', 'multiagent_training')
        logging.info('Policy: {}LSTM-RL {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
        return super().predict(state)

