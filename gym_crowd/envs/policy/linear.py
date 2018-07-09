from gym_crowd.envs.policy.policy import Policy
from gym_crowd.envs.utils.action import ActionXY
import numpy as np


class LinearPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def predict(self, state, kinematics):
        assert kinematics is False

        theta = np.arctan2(state.gy-state.py, state.gx-state.px)
        vx = np.cos(theta) * state.v_pref
        vy = np.sin(theta) * state.v_pref
        action = ActionXY(vx, vy)

        return action
