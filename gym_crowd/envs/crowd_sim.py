import logging
import os
import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import trajnettools
from numpy.linalg import norm
import gym_crowd
from gym_crowd.envs.utils.pedestrian import Pedestrian
from gym_crowd.envs.utils.utils import point_to_segment_dist
from gym_crowd.envs.utils.action import ActionXY


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
        self.navigator = None
        self.peds = None
        self.global_time = None
        self.ped_times = None
        self.states = None
        self.attention_weights = None
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        # orca simulation
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.ped_num = None
        # trajnet simulation
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
            self.case_size['test'] = len(self.scenes['test'])
        else:
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.ped_num = config.getint('sim', 'ped_num')
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

    def set_navigator(self, navigator):
        self.navigator = navigator

    def generate_random_ped_position(self, ped_num, rule):
        """
        Generate ped position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param ped_num:
        :param rule:
        :return:
        """
        if rule == 'square_crossing':
            self.peds = []
            for i in range(ped_num):
                ped = Pedestrian(self.config, 'peds')
                if np.random.random() > 0.5:
                    sign = -1
                else:
                    sign = 1
                while True:
                    px = np.random.random() * self.square_width * 0.5 * sign
                    py = (np.random.random() - 0.5) * self.square_width
                    collide = False
                    for agent in [self.navigator] + self.peds:
                        if norm((px - agent.px, py - agent.py)) < ped.radius + agent.radius:
                            collide = True
                            break
                    if not collide:
                        break
                while True:
                    gx = np.random.random() * self.square_width * 0.5 * -sign
                    gy = (np.random.random() - 0.5) * self.square_width
                    collide = False
                    for agent in [self.navigator] + self.peds:
                        if norm((gx - agent.gx, gy - agent.gy)) < ped.radius + agent.radius:
                            collide = True
                            break
                    if not collide:
                        break
                ped.set(px, py, gx, gy, 0, 0, 0)
                self.peds.append(ped)
        elif rule == 'circle_crossing':
            self.peds = []
            for i in range(ped_num):
                ped = Pedestrian(self.config, 'peds')
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases navigator could meet with pedestrian
                    px_noise = (np.random.random() - 0.5) * ped.v_pref
                    py_noise = (np.random.random() - 0.5) * ped.v_pref
                    px = self.circle_radius * np.cos(angle) + px_noise
                    py = self.circle_radius * np.sin(angle) + py_noise
                    collide = False
                    for agent in [self.navigator] + self.peds:
                        if norm((px - agent.px, py - agent.py)) < ped.radius + agent.radius or \
                                norm((px - agent.gx, py - agent.gy)) < ped.radius + agent.radius:
                            collide = True
                            break
                    if not collide:
                        break
                ped.set(px, py, -px, -py, 0, 0, 0)
                self.peds.append(ped)

    def get_average_ped_time(self):
        if not self.navigator.reached_destination():
            raise ValueError('Episode is not done yet')
        # run simulation until all agents are done
        while not all(self.ped_times):
            self.step(ActionXY(0, 0))
        return sum(self.ped_times) / len(self.ped_times)

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for navigator and peds
        :return:
        """
        if self.navigator is None:
            raise AttributeError('Navigator has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        self.ped_times = [0] * self.ped_num

        if self.config.get('peds', 'policy') == 'trajnet':
            if phase == 'train':
                pass
            elif phase == 'val':
                pass
            else:
                scene_index = test_case if test_case is not None else self.case_counter
                self.case_counter['test'] = (self.case_counter['test'] + 1) % self.case_capacity['test']
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
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.navigator.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            np.random.seed(counter_offset[phase] + self.case_counter[phase])
            if phase in ['train', 'val']:
                ped_num = self.ped_num if self.navigator.policy.multiagent_training else 1
                self.generate_random_ped_position(ped_num=ped_num, rule=self.train_val_sim)
            else:
                self.generate_random_ped_position(ped_num=self.ped_num, rule=self.test_sim)
            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]

        for agent in [self.navigator] + self.peds:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = [[self.navigator.get_full_state(), [ped.get_full_state() for ped in self.peds]]]
        if hasattr(self.navigator.policy, 'get_attention_weights'):
            self.attention_weights = [np.array([0.2] * len(self.peds))]

        # get current observation
        if self.navigator.sensor == 'coordinates':
            ob = [ped.get_observable_state() for ped in self.peds]
        elif self.navigator.sensor == 'RGB':
            raise NotImplemented

        return ob

    def step(self, action):
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
            if self.navigator.kinematics == 'holonomic':
                vx = ped.vx - action.vx
                vy = ped.vy - action.vy
            else:
                vx = ped.vx - action.v * np.cos(action.r + self.navigator.theta)
                vy = ped.vy - action.v * np.sin(action.r + self.navigator.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0)
            if closest_dist < ped.radius + self.navigator.radius:
                collision = True
                # logging.debug("Collision: distance between navigator and p{} is {:.2f}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between pedestrians
        ped_num = len(self.peds)
        for i in range(ped_num):
            for j in range(i+1, ped_num):
                dx = self.peds[i].px - self.peds[j].px
                dy = self.peds[i].py - self.peds[j].py
                dist = (dx**2 + dy**2)**(1/2)                
                if dist < self.peds[i].radius + self.peds[j].radius:
                    collision = True
                    # logging.debug("Collision: distance between p{} and p{} is {:.2f}".format(i, j, dist))

        # check if reaching the goal
        end_position = np.array(self.navigator.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.navigator.get_goal_position())) < self.navigator.radius

        if self.global_time >= self.time_limit-1:
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
            reward = -0.1 + dmin / 2
            done = False
            info = 'too close'
        else:
            reward = 0
            done = False
            info = ''

        # update all agents
        self.navigator.step(action)
        for i, ped_action in enumerate(ped_actions):
            self.peds[i].step(ped_action)
        self.global_time += self.time_step
        for i, ped in enumerate(self.peds):
            # only record the first time the ped reaches the goal
            if self.ped_times[i] == 0 and ped.reached_destination():
                self.ped_times[i] = self.global_time
        self.states.append([self.navigator.get_full_state(), [ped.get_full_state() for ped in self.peds]])
        if hasattr(self.navigator.policy, 'get_attention_weights'):
            self.attention_weights.append(self.navigator.policy.get_attention_weights())

        if self.navigator.sensor == 'coordinates':
            ob = [ped.get_observable_state() for ped in self.peds]
        elif self.navigator.sensor == 'RGB':
            raise NotImplemented

        return ob, reward, done, info

    def render(self, mode='human', output_file=None):
        navigator_color = 'yellow'
        goal_color = 'blue'
        heading_color = 'red'

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            for ped in self.peds:
                ped_circle = plt.Circle(ped.get_position(), ped.radius, fill=True, color='b')
                ax.add_artist(ped_circle)
            ax.add_artist(plt.Circle(self.navigator.get_position(), self.navigator.radius, fill=True, color='r'))
            plt.show()
        if mode == 'traj':
            navigator_positions = [self.states[i][0].position for i in range(len(self.states))]
            ped_positions = [[self.states[i][1][j].position for j in range(len(self.peds))]
                             for i in range(len(self.states))]
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            for k in range(len(self.states)):
                navigator = plt.Circle(navigator_positions[k], self.navigator.radius, fill=True, color=navigator_color)
                peds = [plt.Circle(ped_positions[k][i], self.peds[i].radius, fill=True, color=str((i+1)/20))
                        for i in range(len(self.peds))]
                ax.add_artist(navigator)
                for ped in peds:
                    ax.add_artist(ped)      
            time = plt.text(-1, 6, 'Trajectories', fontsize=12)
            ax.add_artist(time)
            plt.legend([navigator], ['navigator'])
            plt.show()
        elif mode == 'video':
            navigator_positions = [state[0].position for state in self.states]
            ped_positions = [[state[1][j].position for j in range(len(self.peds))] for state in self.states]
            x_offset = 0.11
            y_offset = 0.11

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            goal = plt.Circle((0, 4), 0.05, fill=True, color=goal_color)
            navigator = plt.Circle(navigator_positions[0], self.navigator.radius, fill=True, color=navigator_color)
            # visualize attention weights using the color saturation
            if self.attention_weights is not None:
                peds = [plt.Circle(ped_positions[0][i], self.peds[i].radius, fill=True,
                                   color=(self.attention_weights[0][i], 0, 0)) for i in range(len(self.peds))]
            else:
                peds = [plt.Circle(ped_positions[0][i], self.peds[i].radius, fill=True, color=str((i+1)/20))
                        for i in range(len(self.peds))]
            ped_annotations = [plt.text(peds[i].center[0]-x_offset, peds[i].center[1]-y_offset, str(i), color='white')
                               for i in range(len(self.peds))]
            time = plt.text(0, 6, 'Step: {}'.format(0), fontsize=12)
            if self.attention_weights is not None:
                attention_scores = [plt.text(-6, 6 - 0.5 * i, 'Ped {}: {:.2f}'.format(i, self.attention_weights[0][i]),
                                             fontsize=12) for i in range(len(self.peds))]
            if self.navigator.kinematics == 'unicycle':
                radius = self.navigator.radius
                heading_pos = [((state[0].px, state[0].px + radius * np.cos(state[0].theta)),
                                (state[0].py, state[0].py + radius * np.sin(state[0].theta))) for state in self.states]
                navigator_heading = plt.Line2D(*heading_pos[0], color=heading_color)
                ax.add_artist(navigator_heading)
            ax.add_artist(navigator)
            ax.add_artist(goal)
            ax.add_artist(time)
            for i, ped in enumerate(peds):
                ax.add_artist(ped)
                ax.add_artist(ped_annotations[i])
            plt.legend([navigator, goal], ['navigator', 'goal'])

            def update(frame_num):
                navigator.center = navigator_positions[frame_num]
                for i, ped in enumerate(peds):
                    ped.center = ped_positions[frame_num][i]
                    ped_annotations[i].set_position((ped.center[0]-x_offset, ped.center[1]-y_offset))
                    if self.navigator.kinematics == 'unicycle':
                        navigator_heading.set_xdata(heading_pos[frame_num][0])
                        navigator_heading.set_ydata(heading_pos[frame_num][1])
                    if self.attention_weights is not None:
                        ped.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('Ped {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step*1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=1/self.time_step, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)

            plt.show()
        else:
            raise NotImplemented
