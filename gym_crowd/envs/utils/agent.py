import numpy as np
from gym_crowd.envs.policy.policy_factory import policy_factory
from gym_crowd.envs.utils.action import ActionXY, ActionRot
from gym_crowd.envs.utils.state import ObservableState, FullState
import abc


class Agent(object):
    def __init__(self, config, section):
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        self.kinematics = config.get(section, 'kinematics')
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None

        assert self.kinematics in ['holonomic', 'unicycle']

    def set(self, px, py, gx, gy, vx, vy, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    @abc.abstractmethod
    def act(self, **kwargs):
        """
        Create state object and pass it to policy
        :param ob:
        :return:
        """

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, time=1):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * time
            py = self.py + action.vy * time
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * time
            py = self.py + np.sin(theta) * action.v * time

        return px, py

    def step(self, action):
        self.check_validity(action)
        pos = self.compute_position(action)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta += action.r
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

