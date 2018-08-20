import torch
import torch.nn as nn
import numpy as np
import logging
from gym_crowd.envs.utils.action import ActionRot, ActionXY
from dynav.policy.utils import reward
from dynav.policy.cadrl import CADRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims, lstm_input_dim, lstm_hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_dims[0]), nn.ReLU(),
                                 nn.Linear(mlp_dims[0], mlp_dims[1]), nn.ReLU(),
                                 nn.Linear(mlp_dims[1], 1))
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of peds, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :4]
        ped_state = state[:, :, 4:]
        h0 = torch.zeros(1, size[0], 20)
        c0 = torch.zeros(1, size[0], 20)
        output, (hn, cn) = self.lstm(ped_state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class CadrlLSTM(CADRL):
    def __init__(self):
        super().__init__()

    def configure(self, config):
        self.set_common_parameters(config)
        self.joint_state_dim = 24
        mlp_dims = [int(x) for x in config.get('cadrl_lstm', 'mlp_dims').split(', ')]
        lstm_hidden_dim = config.getint('cadrl_lstm', 'lstm_hidden_dim')
        lstm_input_dim = 7
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims, lstm_input_dim, lstm_hidden_dim)
        self.multiagent_training = config.getboolean('cadrl_lstm', 'multiagent_training')
        logging.info('LSTM: {} agent training'.format('single' if not self.multiagent_training else 'multiple'))

    def predict(self, state):
        """
        Input state is the joint state of navigator concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                batch_next_states = []
                # sort ped order by decreasing distance to the navigator

                def dist(ped):
                    return np.linalg.norm(np.array(ped.position) - np.array(state.self_state.position))
                state.ped_states = sorted(state.ped_states, key=dist, reverse=True)
                for ped_state in state.ped_states:
                    next_self_state = self.propagate(state.self_state, action)
                    next_ped_state = self.propagate(ped_state, ActionXY(ped_state.vx, ped_state.vy))
                    next_dual_state = torch.Tensor([next_self_state + next_ped_state]).to(self.device)
                    batch_next_states.append(next_dual_state)
                batch_next_states = torch.cat(batch_next_states, dim=0)
                # VALUE UPDATE
                next_state_value = self.model(self.rotate(batch_next_states).unsqueeze(0)).data.item()
                gamma_bar = pow(self.gamma, self.time_step * state.self_state.v_pref)
                value = reward(state, action, self.kinematics, self.time_step) + gamma_bar * next_state_value
                if value > max_value:
                    max_value = value
                    max_action = action

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def transform(self, state):
        """
        Take the state passed from agent and transform it to tensor for batch training

        :param state:
        :return: tensor of shape (len(state), )
        """
        state = torch.cat([torch.Tensor([state.self_state + ped_state]).to(self.device)
                          for ped_state in state.ped_states], dim=0)
        state = self.rotate(state)
        return state
