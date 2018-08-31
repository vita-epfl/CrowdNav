import torch
import numpy as np
import logging
from gym_crowd.envs.utils.action import ActionRot, ActionXY
from dynav.policy.sarl import SARL, ValueNetwork


class OmSarl(SARL):
    def __init__(self):
        super().__init__()
        self.cell_num = None
        self.cell_size = None

    def configure(self, config):
        self.set_common_parameters(config)
        self.cell_num = config.getint('om_sarl', 'cell_num')
        self.cell_size = config.getfloat('om_sarl', 'cell_size')

        mlp1_dims = [int(x) for x in config.get('om_sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('om_sarl', 'mlp2_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('om_sarl', 'attention_dims').split(', ')]
        global_state_dim = config.getint('om_sarl', 'global_state_dim')
        self.model = ValueNetwork(self.joint_state_dim, self.ped_state_dim, mlp1_dims, mlp2_dims,
                                  attention_dims, global_state_dim, input_dim=self.joint_state_dim + self.cell_num ** 2)
        self.multiagent_training = config.getboolean('om_sarl', 'multiagent_training')
        logging.info('OM-SARL: {} agent training'.format('single' if not self.multiagent_training else 'multiple'))

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
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                ob, reward, done, info = self.env.onestep_lookahead(action)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_ped_state]).to(self.device)
                                              for next_ped_state in ob], dim=0)
                occupancy_maps = self.build_occupancy_maps(ob, next_self_state)
                rotated_batch_input = torch.cat([self.rotate(batch_next_states), occupancy_maps], dim=1).unsqueeze(0)
                # VALUE UPDATE
                next_state_value = self.model(rotated_batch_input).data.item()
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
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
        occupancy_maps = self.build_occupancy_maps(state.ped_states, state.self_state)
        state = torch.cat([torch.Tensor([state.self_state + ped_state]).to(self.device)
                          for ped_state in state.ped_states], dim=0)
        state = torch.cat([self.rotate(state), occupancy_maps], dim=1)
        return state

    def get_attention_weights(self):
        return self.model.attention_weights

    def build_occupancy_maps(self, ped_states, self_state):
        """

        :param ped_states:
        :param self_state:
        :return: tensor of shape (# ped - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for ped in ped_states:
            other_peds = np.concatenate([np.array([(other_ped.px, other_ped.py, other_ped.vx, other_ped.vy)])
                                         for other_ped in ped_states + [self_state] if other_ped != ped], axis=0)
            other_px = other_peds[:, 0] - ped.px
            other_py = other_peds[:, 1] - ped.py
            # new x-axis is in the direction of ped velocity
            ped_velocity_angle = np.arctan2(ped.vy, ped.vx)
            other_ped_orientation = np.arctan2(other_py, other_px)
            rotation = other_ped_orientation - ped_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # calculate relative velocity for other agents
            # other_ped_velocity_angles = np.arctan2(other_peds[:, 3], other_peds[:, 2])
            # rotation = other_ped_velocity_angles - ped_velocity_angle
            # speed = np.linalg.norm(other_peds[:, 2], other_peds[:, 3])
            # other_peds[:, 2] = np.cos(rotation) * speed
            # other_peds[:, 3] = np.sin(rotation) * speed

            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = 4 * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            occupancy_maps.append([occupancy_map.astype(int)])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()
