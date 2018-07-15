import numpy as np
from gym_crowd.envs.policy.policy import Policy
from gym_crowd.envs.utils.action import ActionXY
from gym_crowd.envs.policy.pyorca.pyorca import ORCAAgent, orca


class ORCA(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def configure(self, config):
        assert True

    def predict(self, state):
        """
        The preferred velocity in pyorca is max speed in the direction of goal, thus a 2D vector
        The max speed in pyorca is the same as v_pref here

        :param state:
        :return:
        """
        if self.reach_destination(state):
            return ActionXY(0, 0)

        tau = 1
        dt = 1

        self_state = state.self_state
        theta = np.arctan2(self_state.gy - self_state.py, self_state.gx - self_state.px)
        vel = np.array((np.cos(theta), np.sin(theta))) * self_state.v_pref
        agent = ORCAAgent((self_state.px, self_state.py), (self_state.vx, self_state.vy),
                          self_state.radius, self_state.v_pref, vel)
        collider = [ORCAAgent((ped.px, ped.py), (ped.vx, ped.vy), ped.radius, None, None)
                    for ped in state.ped_states]
        new_velocity, _ = orca(agent, collider, tau, dt)
        action = ActionXY(new_velocity[0], new_velocity[1])

        return action
