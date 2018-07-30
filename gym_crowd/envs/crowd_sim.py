import logging
import os
import time
import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import trajnettools
import gym_crowd
from gym_crowd.envs.utils.pedestrian import Pedestrian
from gym_crowd.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be pedestrian or navigator.
        Pedestrians are controlled by a unknown and fixed policy.
        Navigator is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
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
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
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

    def generate_random_ped_position(self, ped_num, rule, square_width=None, radius=None):
        """
        Generate ped position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param ped_num:
        :param square_width:
        :param rule:
        :param radius:
        :return:
        """
        assert rule in ['square_crossing', 'circle_crossing']
        if rule == 'square_crossing':
            # TODO: also check other peds
            self.peds = []
            for i in range(ped_num):
                ped = Pedestrian(self.config, 'peds')
                if np.random.random() > 0.5:
                    sign = -1
                else:
                    sign = 1
                while True:
                    px = np.random.random() * square_width * 0.5 * sign
                    py = (np.random.random() - 0.5) * square_width
                    if norm((px - self.navigator.px, py - self.navigator.py)) > \
                            ped.radius + self.navigator.radius:
                        break
                while True:
                    gx = np.random.random() * square_width * 0.5 * -sign
                    gy = (np.random.random() - 0.5) * square_width
                    if norm((gx - self.navigator.gx, gy - self.navigator.gy)) > \
                            ped.radius + self.navigator.radius:
                        break
                ped.set(px, py, gx, gy, 0, 0, 0)
                self.peds.append(ped)
        elif rule == 'circle_crossing':
            assert ped_num == 1
            self.peds = [Pedestrian(self.config, 'peds') for _ in range(ped_num)]
            angle = np.random.uniform(low=-np.pi / 2 + np.arcsin(0.3 / 2) * radius,
                                      high=3 / 2 * np.pi - np.arcsin(0.3 / 2) * radius)
            # add some noise to simulate all the possible cases navigator could meet with pedestrian
            px_noise = (np.random.random() - 0.5) * self.peds[0].v_pref
            py_noise = (np.random.random() - 0.5) * self.peds[0].v_pref
            self.peds[0].set(radius * np.cos(angle) + px_noise, radius * np.sin(angle) + py_noise,
                             radius * np.cos(angle + np.pi), radius * np.sin(angle + np.pi), 0, 0, 0)

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for navigator and peds
        :return:
        """
        if self.navigator is None:
            raise AttributeError('Navigator has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter = test_case
        single_agent_simulation = ['cadrl', 'orca', 'srl']
        multiple_agent_simulation = ['srl', 'orca']
        self.timer = 0

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
                self.navigator.set(0, -4, 0, 4, 0, 0, np.pi / 2)
                for i in range(ped_num):
                    # assign ith trajectory to ith ped's policy
                    self.peds[i].policy.configure(scene[i])
                    self.peds[i].set(scene[i][0].x, scene[i][0].y, scene[i][-1].x, scene[i][-1].y, 0, 0, 0)
        else:
            square_width = 10
            radius = 4
            self.navigator.set(0, -radius, 0, radius, 0, 0, np.pi / 2)
            if phase == 'train':
                np.random.seed(int(time.time()))
                if self.navigator.policy.name in single_agent_simulation:
                    self.generate_random_ped_position(ped_num=1, rule='circle_crossing', radius=radius)
                elif self.navigator.policy.name in multiple_agent_simulation:
                    self.generate_random_ped_position(ped_num=5, rule='square_crossing', square_width=square_width)
                else:
                    raise NotImplemented
            elif phase == 'val':
                np.random.seed(0 + self.case_counter)
                if self.navigator.policy.name in single_agent_simulation:
                    self.generate_random_ped_position(ped_num=1, rule='circle_crossing', radius=radius)
                elif self.navigator.policy.name in multiple_agent_simulation:
                    self.generate_random_ped_position(ped_num=5, rule='square_crossing', square_width=square_width)
                else:
                    raise NotImplemented
                self.case_counter = (self.case_counter + 1) % self.val_size
            else:
                if self.case_counter >= 0:
                    np.random.seed(1000 + self.case_counter)
                    self.generate_random_ped_position(ped_num=5, rule='square_crossing', square_width=square_width)
                    self.case_counter = (self.case_counter + 1) % self.test_size
                else:
                    # for hand-crafted cases
                    if self.case_counter == -1:
                        self.peds = [Pedestrian(self.config, 'peds') for _ in range(6)]
                        self.peds[0].set(-0.75, -3, 1, -3, 0, 0, 0)
                        self.peds[1].set(-1.75, -2, 2, -2, 0, 0, 0)
                        self.peds[2].set(-2.75, -1, 3, -1, 0, 0, 0)
                        self.peds[3].set(-3.75, 0, 4, 0, 0, 0, 0)
                        self.peds[4].set(-4.75, 1, 5, 1, 0, 0, 0)
                        self.peds[5].set(-5.75, 2, 6, 2, 0, 0, 0)

        for agent in [self.navigator] + self.peds:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

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

        # collision detection
        dmin = float('inf')
        collision = False
        for i, ped in enumerate(self.peds):
            px = ped.px - self.navigator.px
            py = ped.py - self.navigator.py
            vx = ped_actions[i].vx - action.vx
            vy = ped_actions[i].vy - action.vy
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0)
            if closest_dist < ped.radius + self.navigator.radius:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if reaching the goal
        end_position = np.array(self.navigator.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.navigator.get_goal_position())) < self.navigator.radius

        if self.timer >= self.time_limit-1:
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
            self.timer += self.time_step
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
            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            navigator = plt.Circle(navigator_positions[0], self.navigator.radius, fill=True, color='red')
            peds = [plt.Circle(ped_positions[0][i], self.peds[i].radius, fill=True, color=str((i+1)/20))
                    for i in range(len(self.peds))]
            text = plt.text(0, 6, 'Step: {}'.format(0), fontsize=12)
            ax.add_artist(text)
            ax.add_artist(navigator)
            for ped in peds:
                ax.add_artist(ped)
            plt.legend([navigator], ['navigator'])

            def update(frame_num):
                navigator.center = navigator_positions[frame_num]
                for i, ped in enumerate(peds):
                    ped.center = ped_positions[frame_num][i]

                text.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step*1000)
            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=1/self.time_step, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)

            plt.show()
        else:
            raise NotImplemented
