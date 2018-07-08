import numpy as np
from gym_crowd.envs.policy.policy_factory import policy_factory
from gym_crowd.envs.utils.action import ActionXY, ActionRot
from gym_crowd.envs.utils.state import State


class Agent(object):
    def __init__(self, config, section):
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        self.kinematics = config.getboolean(section, 'kinematics')
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None

    def set(self, px, py, gx, gy, vx, vy, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

    def get_observable_state(self):
        return self.px, self.py, self.radius, self.theta

    def get_full_state(self):
        return self.px, self.py, self.radius, self.theta

    def act(self, ob):
        """
        Create state object and pass it to policy
        :param ob:
        :return:
        """
        state = State(self.px, self.py, self.gx, self.gy, self.v_pref, self.radius, ob)
        action = self.policy.predict(state, self.kinematics)

        return action

    def check_validity(self, action):
        if self.kinematics:
            assert isinstance(action, ActionRot)
        else:
            assert isinstance(action, ActionXY)

    def compute_position(self, action, time=1):
        self.check_validity(action)
        if self.kinematics:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * time
            py = self.py + np.sin(theta) * action.v * time
        else:
            px = self.px + action.vx * time
            py = self.py + action.vy * time

        return px, py

    def step(self, action):
        self.check_validity(action)
        pos = self.compute_position(action)
        self.px, self.py = pos
        if self.kinematics:
            self.theta += action.r
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)
        else:
            self.vx = action.vx
            self.vy = action.vy
