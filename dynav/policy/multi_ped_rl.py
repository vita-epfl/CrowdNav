import torch
import numpy as np
from gym_crowd.envs.utils.action import ActionRot, ActionXY
from dynav.policy.cadrl import CADRL


class MultiPedRL(CADRL):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # peds, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                ob, reward, done, info = self.env.onestep_lookahead(action)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_ped_state]).to(self.device)
                                              for next_ped_state in ob], dim=0)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(ob).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)
                # VALUE UPDATE
                next_state_value = self.model(rotated_batch_input).data.item()
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of peds, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.self_state + ped_state]).to(self.device)
                                  for ped_state in state.ped_states], dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.ped_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + self.cell_num ** 2 * self.om_channel_size if self.with_om else self.joint_state_dim

    def build_occupancy_maps(self, ped_states):
        """

        :param ped_states:
        :return: tensor of shape (# ped - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for ped in ped_states:
            other_peds = np.concatenate([np.array([(other_ped.px, other_ped.py, other_ped.vx, other_ped.vy)])
                                         for other_ped in ped_states if other_ped != ped], axis=0)
            other_px = other_peds[:, 0] - ped.px
            other_py = other_peds[:, 1] - ped.py
            # new x-axis is in the direction of ped's velocity
            ped_velocity_angle = np.arctan2(ped.vy, ped.vx)
            other_ped_orientation = np.arctan2(other_py, other_px)
            rotation = other_ped_orientation - ped_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of peds in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_ped_velocity_angles = np.arctan2(other_peds[:, 3], other_peds[:, 2])
                rotation = other_ped_velocity_angles - ped_velocity_angle
                speed = np.linalg.norm(other_peds[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[2 * int(index)].append(1)
                            dm[2 * int(index) + 1].append(other_vx[i])
                            dm[2 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplemented
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

