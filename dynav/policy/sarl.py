import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import itertools
from gym_crowd.envs.policy.policy import Policy
from gym_crowd.envs.utils.action import ActionRot, ActionXY
from gym_crowd.envs.utils.state import ObservableState, FullState
from dynav.policy.utils import reward


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, kinematics, mlp1_dims, mlp2_dims):
        super().__init__()
        self.input_dim = input_dim
        self.kinematics = kinematics
        self.mlp1 = nn.Sequential(nn.Linear(input_dim, mlp1_dims[0]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[0], mlp1_dims[1]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[1], mlp1_dims[2]), nn.ReLU(),
                                  nn.Linear(mlp1_dims[2], mlp2_dims))
        self.mlp2 = nn.Sequential(nn.Linear(mlp2_dims, 1))
        self.attention = nn.Sequential(nn.Linear(mlp2_dims, 1))

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
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)

        new_state = torch.cat([dg, v_pref, vx, vy, radius, theta, vx1, vy1, px1, py1, radius1, radius_sum,
                               cos_theta, sin_theta, da], dim=1)
        return new_state

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of peds, length of a joint state)
        :return:
        """
        size = state.shape
        state = self.rotate(torch.reshape(state, (-1, size[2])))
        mlp1_output = self.mlp1(state)
        scores = torch.reshape(self.attention(mlp1_output), (size[0], size[1], 1)).squeeze(dim=2)
        weights = softmax(scores, dim=1).unsqueeze(2)
        features = torch.reshape(mlp1_output, (size[0], size[1], -1))
        weighted_feature = torch.sum(weights.expand_as(features) * features, dim=1)
        value = self.mlp2(weighted_feature)
        return value


class SARL(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.discrete = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.action_space_size = None
        self.action_space = None

    def configure(self, config):
        self.gamma = config.getfloat('rl', 'gamma')

        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.action_space_size = config.getint('action_space', 'action_space_size')
        self.discrete = config.getboolean('action_space', 'discrete')

        input_dim = config.getint('sarl', 'input_dim')
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = config.getint('sarl', 'mlp2_dims')
        self.model = ValueNetwork(input_dim, self.kinematics, mlp1_dims, mlp2_dims)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')

        assert self.action_space_size in [50, 100]
        assert self.sampling in ['uniform', 'exponential']
        assert self.kinematics in ['holonomic', 'unicycle']

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        if self.kinematics == 'holonomic':
            if self.action_space is not None:
                return self.action_space
            if self.discrete:
                speed_grids, rotation_grids = (10, 10) if self.action_space_size == 100 else (8, 6)
            else:
                speed_grids, rotation_grids = (8, 6) if self.action_space_size == 100 else (5, 5)

            action_space = [ActionXY(0, 0)]
            if self.sampling == 'exponential':
                speeds = [(np.exp((i + 1) / speed_grids) - 1) / (np.e - 1) * v_pref for i in range(speed_grids)]
            else:
                speeds = [(i + 1) / speed_grids * v_pref for i in range(speed_grids)]
            rotations = [i / rotation_grids * 2 * np.pi for i in range(rotation_grids)]
            for speed, rotation in itertools.product(speeds, rotations):
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))

            if self.discrete:
                # always recompute action space for continuous action space
                for i in range(int(self.action_space_size / 2)):
                    random_speed = np.random.random() * v_pref
                    random_rotation = np.random.random() * 2 * np.pi
                    action_space.append(ActionXY(random_speed * np.cos(random_rotation),
                                                 random_speed * np.sin(random_rotation)))
            else:
                self.action_space = action_space
        else:
            raise NotImplemented

        return action_space

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of peds
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, state.vx, state.vy, state.radius,
                                       state.gx, state.gy, state.v_pref, state.theta)
            else:
                next_px = state.px + np.cos(action.r + state.theta) * action.v * self.time_step
                next_py = state.py + np.sin(action.r + state.theta) * action.v * self.time_step
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                       state.v_pref, next_theta)
        else:
            raise ValueError('Type error')

        return next_state

    def predict(self, state):
        """
        Input state is the joint state of navigator concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0)
        action_space = self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = action_space[np.random.choice(len(action_space))]
        else:
            max_value = float('-inf')
            max_action = None
            for action in action_space:
                batch_next_states = []
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
        :return: tensor of shape (# of peds, len(state))
        """
        return torch.cat([torch.Tensor([state.self_state + ped_state]).to(self.device)
                          for ped_state in state.ped_states], dim=0)
