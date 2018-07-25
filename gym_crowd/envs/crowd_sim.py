import os
import gym_crowd
import gym
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import trajnettools
from gym_crowd.envs.utils.pedestrian import Pedestrian


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be pedestrian or navigator.
        Pedestrians are controlled by a unknown and fixed policy.
        Navigator is controlled by a known and learnable policy.

        """
        self.train_ped_num = None
        self.time_limit = None
        self.peds = None
        self.navigator = None
        self.timer = None
        self.states = None
        self.config = None
        self.test_size = None
        self.val_size = None
        self.case_counter = None
        self.scenes = None

    def configure(self, config):
        self.config = config
        self.train_ped_num = config.getint('env', 'train_ped_num')
        self.time_limit = config.getint('env', 'time_limit')
        if self.config.get('peds', 'policy') == 'trajnet':
            # load trajnet data
            trajnet_dir = os.path.join(os.path.dirname(gym_crowd.__file__), 'envs/data/trajnet')
            train_dir = os.path.join(trajnet_dir, 'train')
            val_dir = os.path.join(trajnet_dir, 'val')
            self.scenes = dict({'train': list(trajnettools.load_all(os.path.join(train_dir, 'biwi_hotel.ndjson'),
                                                                    as_paths=True, sample={'syi.ndjson': 0.05})),
                                'val': list(trajnettools.load_all(os.path.join(val_dir, 'biwi_hotel.ndjson'),
                                                                  as_paths=True, sample={'syi.ndjson': 0.05})),
                                'test': list(trajnettools.load_all(os.path.join(val_dir, 'wildtrack.ndjson'),
                                                                   as_paths=True, sample={'syi.ndjson': 0.05}))})
            for phase in ['train', 'val', 'test']:
                logging.info('Number of scenes in phase {}: {}'.format(phase.upper(), len(self.scenes[phase])))
            self.test_size = len(self.scenes['test'])
        else:
            self.val_size = 100
            self.test_size = 100
        self.case_counter = 0

    def set_navigator(self, navigator):
        self.navigator = navigator

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for navigator and peds
        :return:
        """
        if self.navigator is None:
            raise AttributeError('Navigator has to be set!')
        self.timer = 0

        assert phase in ['train', 'val', 'test']
        if self.config.get('peds', 'policy') == 'trajnet':
            if phase == 'train':
                pass
            elif phase == 'val':
                pass
            else:
                scene_index = test_case if test_case is not None else self.case_counter
                self.case_counter = (self.case_counter + 1) % self.test_size
                scene = self.scenes[phase][scene_index][1]
                ped_num = len(scene)
                if test_case is not None:
                    logging.info('{} pedestrians in scene {}'.format(ped_num, scene_index))
                self.peds = [Pedestrian(self.config, 'peds') for _ in range(ped_num)]
                self.navigator.set(0, -2, 0, 2, 0, 0, np.pi / 2)
                for i in range(ped_num):
                    # assign ith trajectory to ith ped's policy
                    self.peds[i].policy.configure(scene[i])
                    self.peds[i].set(scene[i][0].x, scene[i][0].y, scene[i][-1].x, scene[i][-1].y, 0, 0, 0)
        else:
            if phase == 'train':
                np.random.seed(int(time.time()))
                self.peds = [Pedestrian(self.config, 'peds') for _ in range(self.train_ped_num)]
                self.navigator.set(0, -2, 0, 2, 0, 0, np.pi / 2)
                angle = np.random.uniform(low=-np.pi/2+np.arcsin(0.3/2)*2, high=3/2*np.pi-np.arcsin(0.3/2)*2)
                # add some random noise
                self.peds[0].set(2*np.cos(angle), 2*np.sin(angle), 2*np.cos(angle+np.pi), 2*np.sin(angle+np.pi), 0, 0, 0)
            elif phase == 'val':
                np.random.seed(0 + self.case_counter)
                self.peds = [Pedestrian(self.config, 'peds') for _ in range(self.train_ped_num)]
                self.navigator.set(0, -2, 0, 2, 0, 0, np.pi / 2)
                angle = np.random.uniform(low=-np.pi/2+np.arcsin(0.3/2)*2, high=3/2*np.pi-np.arcsin(0.3/2)*2)
                self.peds[0].set(2*np.cos(angle), 2*np.sin(angle), 2*np.cos(angle+np.pi), 2*np.sin(angle+np.pi), 0, 0, 0)
                self.case_counter = (self.case_counter + 1) % self.val_size
            else:
                if test_case == 0 or (test_case is None and self.case_counter == 0):
                    self.navigator.set(0, -2, 0, 2, 0, 0, np.pi/2)
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(2)]
                    self.peds[0].set(-1, -1, 1, -1, 0, 0, 0)
                    self.peds[1].set(-1, 1, 1, 1, 0, 0, 0)
                elif test_case == 1 or (test_case is None and self.case_counter == 1):
                    self.navigator.set(0, -2, 0, 2, 0, 0, np.pi/2)
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(6)]
                    self.peds[0].set(-1, -1, 1, -1, 0, 0, 0)
                    self.peds[1].set(-1, 0, 1, 0, 0, 0, 0)
                    self.peds[2].set(-1, 1, 1, 1, 0, 0, 0)
                    self.peds[3].set(1, -1, -1, -1, 0, 0, 0)
                    self.peds[4].set(1, 0, -1, 0, 0, 0, 0)
                    self.peds[5].set(1, 1, -1, 1, 0, 0, 0)
                elif test_case == 2 or (test_case is None and self.case_counter == 2):
                    self.navigator.set(0, -2, 0, 2, 0, 0, np.pi/2)
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(4)]
                    self.peds[0].set(-1, -1, 1, 1, 0, 0, 0)
                    self.peds[1].set(-1, 1, 1, -1, 0, 0, 0)
                    self.peds[2].set(1, 1, -1, -1, 0, 0, 0)
                    self.peds[3].set(1, -1, -1, 1, 0, 0, 0)
                elif test_case == 3 or (test_case is None and self.case_counter == 3):
                    self.navigator.set(0, -2, 0, 2, 0, 0, np.pi / 2)
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(3)]
                    self.peds[0].set(-1, 1, -1, -1, 0, 0, 0)
                    self.peds[1].set(0, 1, 0, -1, 0, 0, 0)
                    self.peds[2].set(1, 1, 1, -1, 0, 0, 0)
                elif test_case == 4 or (test_case is None and self.case_counter == 4):
                    self.navigator.set(0, -2, 0, 2, 0, 0, np.pi / 2)
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(3)]
                    self.peds[0].set(-1, 1, 1, -1, 0, 0, 0)
                    self.peds[1].set(-1, 2, 1, 0, 0, 0, 0)
                    self.peds[2].set(-1, 0, 1, -2, 0, 0, 0)
                else:
                    np.random.seed(1000 + self.case_counter)
                    self.navigator.set(0, -2, 0, 2, 0, 0, np.pi / 2)
                    self.peds = []
                    for i in range(5):
                        ped = Pedestrian(self.config, 'peds')
                        ped.set(np.random.random()*5, np.random.random()*5, np.random.random()*5,
                                np.random.random()*5, 0, 0, 0)
                        self.peds.append(ped)
                self.case_counter = (self.case_counter + 1) % self.test_size

        self.states = [[self.navigator.get_full_state(), [ped.get_full_state() for ped in self.peds]]]

        # get current observation
        if self.navigator.sensor == 'coordinates':
            ob = [ped.get_observable_state() for ped in self.peds]
        elif self.navigator.sensor == 'RGB':
            raise NotImplemented

        return ob

    def reward(self, action):
        """
        Only compute reward but don't update the state.

        """
        _, reward, _, _ = self.step(action, update=False)
        return reward

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
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
            for t in np.arange(0, 1.001, 0.1):
                pos = self.navigator.compute_position(action, t)
                ped_pos = ped.compute_position(ped_actions[i], t)
                distance = np.linalg.norm((pos[0]-ped_pos[0], pos[1]-ped_pos[1])) - ped.radius - self.navigator.radius
                if distance < 0:
                    collision = True
                    break
                else:
                    dmin = distance
        reaching_goal = np.linalg.norm((pos[0]-self.navigator.gx, pos[1]-self.navigator.gy)) < self.navigator.radius
        if self.timer >= self.time_limit:
            reward = 0
            done = True
            info = 'timeout'
        elif collision:
            reward = -0.25
            done = True
            info = 'collision'
        elif reaching_goal:
            reward = 1
            done = True
            info = 'reach goal'
        elif dmin < 0.2:
            reward = -0.1 - dmin / 2
            done = False
            info = 'too close'
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
            self.states.append([self.navigator.get_full_state(), [ped.get_full_state() for ped in self.peds]])

        if self.navigator.sensor == 'coordinates':
            ob = [ped.get_observable_state() for ped in self.peds]
        elif self.navigator.sensor == 'RGB':
            raise NotImplemented

        return ob, reward, done, info

    def render(self, mode='human', output_file=None):
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            for ped in self.peds:
                ped_circle = plt.Circle(ped.get_position(), ped.radius, fill=True, color='b')
                ax.add_artist(ped_circle)
            ax.add_artist(plt.Circle(self.navigator.get_position(), self.navigator.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'video':
            navigator_positions = [self.states[i][0].position for i in range(len(self.states))]
            ped_positions = [[self.states[i][1][j].position for j in range(len(self.peds))]
                             for i in range(len(self.states))]

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            navigator = plt.Circle(navigator_positions[0], self.navigator.radius, fill=True, color='red')
            peds = [plt.Circle(ped_positions[0][i], self.peds[i].radius, fill=True, color=str((i+1)/20))
                    for i in range(len(self.peds))]
            text = plt.text(0, 8, 'Step: {}'.format(0), fontsize=12)
            ax.add_artist(navigator)
            for ped in peds:
                ax.add_artist(ped)
            ax.add_artist(text)
            plt.legend([navigator], ['navigator'])

            def update(frame_num):
                navigator.center = navigator_positions[frame_num]
                for i, ped in enumerate(peds):
                    ped.center = ped_positions[frame_num][i]

                text.set_text('Step: {}'.format(frame_num))

            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=400)
            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=2.5, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)

            plt.show()
        else:
            raise NotImplemented
