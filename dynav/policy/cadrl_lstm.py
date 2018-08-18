import torch
import torch.nn as nn
import numpy as np
import logging
from gym_crowd.envs.utils.action import ActionRot, ActionXY
from dynav.policy.utils import reward
from dynav.policy.cadrl import CADRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, kinematics, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.kinematics = kinematics
        self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_dims[0]), nn.ReLU(),
                                 nn.Linear(mlp_dims[0], mlp_dims[1]), nn.ReLU(),
                                 nn.Linear(mlp_dims[1], 1))
        self.lstm = nn.LSTM(7, lstm_hidden_dim, batch_first=True)

    def rotate(self, state):
        """

        :param state: tensor of size (batch_size, # of peds, length of joint state)
        :return:
        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            theta = (state[:, 8]).reshape((batch, -1))
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1)) - vx
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1)) - vy
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)

        self_state = torch.cat([dg, v_pref, theta, radius], dim=1)
        ped_state = torch.cat([px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

        return self_state, ped_state

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of peds, length of a joint state)
        :return:
        """
        size = state.shape
        self_state, ped_state = self.rotate(torch.reshape(state, (-1, size[2])))
        self_state = torch.reshape(self_state, (size[0], size[1], -1))[:, 0, :]
        ped_state = torch.reshape(ped_state, (size[0], size[1], -1))
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
        self.gamma = config.getfloat('rl', 'gamma')

        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')

        input_dim = config.getint('cadrl_lstm', 'input_dim')
        mlp_dims = [int(x) for x in config.get('cadrl_lstm', 'mlp_dims').split(', ')]
        lstm_hidden_dim = config.getint('cadrl_lstm', 'lstm_hidden_dim')
        self.model = ValueNetwork(input_dim, self.kinematics, mlp_dims, lstm_hidden_dim)
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
            return ActionXY(0, 0)
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
                batch_next_states = torch.cat(batch_next_states, dim=0).unsqueeze(0)
                value = reward(state, action, self.kinematics, self.time_step) + \
                    pow(self.gamma, state.self_state.v_pref) * self.model(batch_next_states).data.item()
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
        return torch.cat([torch.Tensor([state.self_state + ped_state]).to(self.device)
                          for ped_state in state.ped_states], dim=0)
