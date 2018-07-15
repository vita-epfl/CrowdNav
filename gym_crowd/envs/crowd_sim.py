import gym
import numpy as np
import matplotlib.pyplot as plt
from gym_crowd.envs.utils.pedestrian import Pedestrian


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Agents consist of pedestrians and navigator.
        Pedestrians are controlled by a unknown and fixed policy.
        Navigator is controlled by a known and learnable policy.

        """
        self.num_peds = None
        self.time_limit = None
        self.peds = None
        self.navigator = None
        self.timer = None

    def configure(self, config):
        self.num_peds = config.getint('env', 'num_peds')
        self.time_limit = config.getint('env', 'time_limit')
        self.peds = [Pedestrian(config, 'peds') for _ in range(self.num_peds)]

    def set_navigator(self, navigator):
        self.navigator = navigator

    def reset(self, phase='test'):
        """
        Set start and goal positions for all agents
        :return:
        """
        self.timer = 0

        # set ped positions
        assert self.num_peds == 2
        self.peds[0].set(-1, -1, 1, -1, 0, 0, 0)
        self.peds[1].set(-1, 1, 1, 1, 0, 0, 0)

        # set navigator position
        self.navigator.set(0, -2, 0, 2, 0, 0, np.pi/2)

        # get current observation
        if self.navigator.sensor == 'coordinates':
            ob = [ped.get_observable_state() for ped in self.peds]
        elif self.navigator.sensor == 'RGB':
            raise NotImplemented

        return ob

    def reward(self, action):
        _, reward, _, _ = self.step(action, update=False)
        return reward

    def step(self, action, update=True):
        ped_actions = []
        for ped in self.peds:
            # observation for peds is always coordinates
            ob = [other_ped.get_observable_state() for other_ped in self.peds if other_ped != ped]
            if self.navigator.visible:
                ob += [self.navigator.get_observable_state()]
            ped_actions.append(ped.act(ob))

        # collision detection for navigator, peds are guaranteed to have no collision
        # TODO: more advanced version of collision detection
        dmin = float('inf')
        collision = False
        for i, ped in enumerate(self.peds):
            if collision:
                break
            for time in np.arange(0, 1.001, 0.25):
                pos = self.navigator.compute_position(action, time)
                ped_pos = ped.compute_position(ped_actions[i], time)
                distance = np.linalg.norm((pos[0]-ped_pos[0], pos[1]-ped_pos[1])) - ped.radius - self.navigator.radius
                if distance < 0:
                    collision = True
                    break
                else:
                    dmin = distance
        reaching_goal = np.linalg.norm((pos[0]-self.navigator.gx, pos[1]-self.navigator.gy)) < self.navigator.radius
        if collision:
            reward = -0.25
            done = True
            info = 'collision'
        elif dmin < 0.2:
            reward = -0.1 - dmin / 2
            done = False
            info = 'to close'
        elif reaching_goal:
            reward = 1
            done = True
            info = 'reaching goal'
        elif self.timer >= self.time_limit:
            reward = 0
            done = True
            info = 'overtime'
        else:
            reward = 0
            done = False
            info = ''

        # update all agents
        if update:
            self.navigator.step(action)
            for i, ped_action in enumerate(ped_actions):
                self.peds[i].step(ped_action)
            self.timer += 1

        if self.navigator.sensor == 'coordinates':
            ob = [ped.get_observable_state() for ped in self.peds]
        elif self.navigator.sensor == 'RGB':
            raise NotImplemented

        return ob, reward, done, info

    def render(self, mode='human', close=False):
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            for ped in self.peds:
                ped_circle = plt.Circle((ped.px, ped.py), ped.radius, fill=True, color='b')
                ax.add_artist(ped_circle)
            ax.add_artist(plt.Circle((self.navigator.px, self.navigator.py), self.navigator.radius, fill=True, color='r'))
            plt.show()
        else:
            raise NotImplemented
