import numpy as np
from numpy.linalg import norm
import abc
import logging
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = config.getboolean(section, 'visible')
        self.v_pref = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

        # Uncertainty for out of sight
        self.uncertainty = 0
        self.last_px = self.px
        self.last_py = self.py
        self.last_vx = self.vx
        self.last_vy = self.vy
        self.last_theta = self.theta

        self.unseen_mode = config.get(section,'unseen_mode')

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None, uncertainty=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref
        if uncertainty is not None:
            self.uncertainty = uncertainty
        else:
            # Robot knows the initial position of each human
            # todo: make more realistic
            self.last_px = self.px
            self.last_py = self.py
            self.last_vx = self.vx
            self.last_vy = self.vy
            self.last_theta = self.theta
    # todo: visualize all these
    def get_observable_state(self):
        if self.unseen_mode == 'ground_truth':
            # always return the real position of agent
            return ObservableState(self.px, self.py, self.vx, self.vy, self.radius, self.uncertainty)
        if self.unseen_mode == 'stationary':
            # always return the last seen position and velocity of the agent
            return ObservableState(self.last_px, self.last_py, self.last_vx, self.last_vy, self.radius, self.uncertainty)
        if self.unseen_mode == 'continuing':
            # assume that the agent keeps its trajectory with the same speed
            if self.uncertainty:
                self.last_px += self.last_vx*self.time_step
                self.last_py += self.last_vy*self.time_step
            return ObservableState(self.last_px, self.last_py, self.last_vx, self.last_vy, self.radius, self.uncertainty)
        if self.unseen_mode == 'slowing_down':
            # assume that the agent slows down as it stays out of view
            if self.uncertainty:
                decay_rate = 0.9
                self.last_vx *= decay_rate
                self.last_vy *= decay_rate
                self.last_px += self.last_vx*self.time_step
                self.last_py += self.last_vy*self.time_step
            return ObservableState(self.last_px, self.last_py, self.last_vx, self.last_vy, self.radius, self.uncertainty)
        if self.unseen_mode == 'expanding_stationary_bubble':
            # assume that the agent cover an increasing area as it stays out of sight
            expansion_rate = 0.5
            radius = self.radius*(1+expansion_rate*self.uncertainty)
            return ObservableState(self.last_px, self.last_py, self.last_vx, self.last_vy, radius, self.uncertainty)
        if self.unseen_mode == 'expanding_moving_bubble':
            # assume that the agent is an ever-growing bubble, moving in the same direction
            expansion_rate = 0.5
            radius = self.radius
            if self.uncertainty:
                self.last_px += self.last_vx*self.time_step
                self.last_py += self.last_vy*self.time_step
                radius = self.radius*(1+expansion_rate*self.uncertainty)
            return ObservableState(self.last_px, self.last_py, self.last_vx, self.last_vy, radius, self.uncertainty)
        if self.unseen_mode == 'enchanced':
            # todo: save last 2 two steps. also add curvature to the assumed path by calculating angular speed.
            raise NotImplementedError

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius, self.uncertainty)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    def get_uncertainty(self):
        return self.uncertainty

    def increment_uncertainty(self,mode='logarithmic',incrementation=1):
        if mode == 'reset':
            self.uncertainty = 0
            return
        if mode == 'linear':
            self.uncertainty += incrementation
            return
        if mode == 'exponential':
            self.uncertainty += incrementation**2 + 2*np.sqrt(self.uncertainty)
            return
        if mode == 'logarithmic':
            self.uncertainty = np.log(incrementation+np.exp(self.uncertainty))
            return

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)
        if not self.uncertainty:
            self.last_px = self.px
            self.last_py = self.py
            self.last_vx = self.vx
            self.last_vy = self.vy
            self.last_theta = self.theta

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

