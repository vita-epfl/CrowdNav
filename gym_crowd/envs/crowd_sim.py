import logging
import gym
import numpy as np
from numpy.linalg import norm
import rvo2
from gym_crowd.envs.utils.pedestrian import Pedestrian
from gym_crowd.envs.utils.utils import point_to_segment_dist
from gym_crowd.envs.utils.info import *


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
        self.action_values = None
        self.attention_weights = None
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
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
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if self.config.get('peds', 'policy') == 'trajnet':
            raise NotImplemented
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

        logging.info('Pedestrian number: {}'.format(self.ped_num))
        if self.randomize_attributes:
            logging.info("Randomize pedestrian's radius and preferred speed")
        else:
            logging.info("Not randomize pedestrian's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

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
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.peds = []
            for i in range(ped_num):
                ped = Pedestrian(self.config, 'peds')
                if self.randomize_attributes:
                    ped.sample_random_attributes()
                if np.random.random() > 0.5:
                    sign = -1
                else:
                    sign = 1
                while True:
                    px = np.random.random() * self.square_width * 0.5 * sign
                    py = (np.random.random() - 0.5) * self.square_width
                    collide = False
                    for agent in [self.navigator] + self.peds:
                        if norm((px - agent.px, py - agent.py)) < ped.radius + agent.radius + self.discomfort_dist:
                            collide = True
                            break
                    if not collide:
                        break
                while True:
                    gx = np.random.random() * self.square_width * 0.5 * -sign
                    gy = (np.random.random() - 0.5) * self.square_width
                    collide = False
                    for agent in [self.navigator] + self.peds:
                        if norm((gx - agent.gx, gy - agent.gy)) < ped.radius + agent.radius + self.discomfort_dist:
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
                if self.randomize_attributes:
                    ped.sample_random_attributes()
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases navigator could meet with pedestrian
                    px_noise = (np.random.random() - 0.5) * ped.v_pref
                    py_noise = (np.random.random() - 0.5) * ped.v_pref
                    px = self.circle_radius * np.cos(angle) + px_noise
                    py = self.circle_radius * np.sin(angle) + py_noise
                    collide = False
                    for agent in [self.navigator] + self.peds:
                        min_dist = ped.radius + agent.radius + self.discomfort_dist
                        if norm((px - agent.px, py - agent.py)) < min_dist or \
                                norm((px - agent.gx, py - agent.gy)) < min_dist:
                            collide = True
                            break
                    if not collide:
                        break
                ped.set(px, py, -px, -py, 0, 0, 0)
                self.peds.append(ped)

    def get_ped_times(self):
        """
        Run the whole simulation to the end and compute the average time for ped to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all peds
        if not self.navigator.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.navigator.get_position(), *params, self.navigator.radius, self.navigator.v_pref,
                     self.navigator.get_velocity())
        for ped in self.peds:
            sim.addAgent(ped.get_position(), *params, ped.radius, ped.v_pref, ped.get_velocity())

        max_time = 1000
        while not all(self.ped_times):
            for i, agent in enumerate([self.navigator] + self.peds):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, ped in enumerate(self.peds):
                if self.ped_times[i] == 0 and ped.reached_destination():
                    self.ped_times[i] = self.global_time

            # for visualization
            self.navigator.set_position(sim.getAgentPosition(0))
            for i, ped in enumerate(self.peds):
                ped.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.navigator.get_full_state(), [ped.get_full_state() for ped in self.peds]])

        del sim
        return self.ped_times

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
        if phase == 'test':
            self.ped_times = [0] * self.ped_num
        else:
            self.ped_times = [0] * (self.ped_num if self.navigator.policy.multiagent_training else 1)
        if not self.navigator.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('peds', 'policy') == 'trajnet':
            raise NotImplemented
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.navigator.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    ped_num = self.ped_num if self.navigator.policy.multiagent_training else 1
                    self.generate_random_ped_position(ped_num=ped_num, rule=self.train_val_sim)
                else:
                    self.generate_random_ped_position(ped_num=self.ped_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    self.ped_num = 3
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(self.ped_num)]
                    self.peds[0].set(0, -6, 0, 5, 0, 0, np.pi/2)
                    self.peds[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.peds[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                elif self.case_counter[phase] == -2:
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(self.ped_num)]
                    self.peds[0].set(-3, 0, -3, 5, 0, 0, np.pi / 2)
                    self.peds[1].set(-2, 0, -2, 5, 0, 0, np.pi / 2)
                    self.peds[2].set(-3, 1, -3, 5, 0, 0, np.pi / 2)
                elif self.case_counter[phase] == -3:
                    self.ped_num = 5
                    self.navigator.set(2.0191, 0.7173, 0.0000, 4.0000, 0.0000, 0.1289, 1.5708)
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(self.ped_num)]
                    self.peds[0].set(-0.1247, 1.4739, 3.3379, 2.8420, 0.6149, 0.7886, 0.0000)
                    self.peds[1].set(1.8997, -0.3898, 3.6275, -0.3209, 0.9992, 0.0399, 0.0000)
                    self.peds[2].set(-0.0124, 0.8459, 2.3581, 3.3641, 0.6854, 0.7281, 0.0000)
                    self.peds[3].set(-0.3289, -0.1324, -4.3308, -1.2543, -0.9629, -0.2699, 0.0000)
                    self.peds[4].set(0.6299, 0.0899, 1.7150, -3.9515, 0.2593, -0.9658, 0.0000)
                elif self.case_counter[phase] == -4:
                    self.ped_num = 5
                    self.peds = [Pedestrian(self.config, 'peds') for _ in range(self.ped_num)]
                    self.peds[0].set(-2, -3, 1, -3, 0, 0, np.pi / 2)
                    self.peds[1].set(3, -2, -3, -2, 0, 0, np.pi / 2)
                    self.peds[2].set(-4, -1, 3, -1, 0, 0, np.pi / 2)
                    self.peds[3].set(5, 0, -5, 0, 0, 0, np.pi / 2)
                    self.peds[4].set(-6, 1, 5, 1, 0, 0, np.pi / 2)
                else:
                    raise NotImplemented

        for agent in [self.navigator] + self.peds:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.navigator.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.navigator.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.navigator.sensor == 'coordinates':
            ob = [ped.get_observable_state() for ped in self.peds]
        elif self.navigator.sensor == 'RGB':
            raise NotImplemented

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

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
            if self.navigator.kinematics == 'holonomic':
                vx = ped.vx - action.vx
                vy = ped.vy - action.vy
            else:
                vx = ped.vx - action.v * np.cos(action.r + self.navigator.theta)
                vy = ped.vy - action.v * np.sin(action.r + self.navigator.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - ped.radius - self.navigator.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between navigator and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between pedestrians
        ped_num = len(self.peds)
        for i in range(ped_num):
            for j in range(i+1, ped_num):
                dx = self.peds[i].px - self.peds[j].px
                dy = self.peds[i].py - self.peds[j].py
                dist = (dx**2 + dy**2)**(1/2) - self.peds[i].radius - self.peds[j].radius
                if dist < 0:
                    # detect collision but don't take peds' collision into account
                    logging.debug('Collision happens between pedestrians in step()')

        # check if reaching the goal
        end_position = np.array(self.navigator.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.navigator.get_goal_position())) < self.navigator.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif self.navigator.visible and dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update:
            # store state, action value and attention weights
            self.states.append([self.navigator.get_full_state(), [ped.get_full_state() for ped in self.peds]])
            if hasattr(self.navigator.policy, 'action_values'):
                self.action_values.append(self.navigator.policy.action_values)
            if hasattr(self.navigator.policy, 'get_attention_weights'):
                self.attention_weights.append(self.navigator.policy.get_attention_weights())

            # update all agents
            self.navigator.step(action)
            for i, ped_action in enumerate(ped_actions):
                self.peds[i].step(ped_action)
            self.global_time += self.time_step
            for i, ped in enumerate(self.peds):
                # only record the first time the ped reaches the goal
                if self.ped_times[i] == 0 and ped.reached_destination():
                    self.ped_times[i] = self.global_time

            # compute the observation
            if self.navigator.sensor == 'coordinates':
                ob = [ped.get_observable_state() for ped in self.peds]
            elif self.navigator.sensor == 'RGB':
                raise NotImplemented
        else:
            if self.navigator.sensor == 'coordinates':
                ob = [ped.get_next_observable_state(action) for ped, action in zip(self.peds, ped_actions)]
            elif self.navigator.sensor == 'RGB':
                raise NotImplemented

        return ob, reward, done, info

    def render(self, mode='human', output_file=None):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        navigator_color = 'yellow'
        goal_color = 'blue'
        heading_color = 'red'
        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', self.ped_num+1)

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
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            for k in range(len(self.states)):
                if k % 2 == 0:
                    navigator = plt.Circle(navigator_positions[k], self.navigator.radius, fill=True, color=navigator_color)
                    peds = [plt.Circle(ped_positions[k][i], self.peds[i].radius, fill=False, color=cmap(i))
                            for i in range(len(self.peds))]
                    ax.add_artist(navigator)
                    for ped in peds:
                        ax.add_artist(ped)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0:
                    agents = peds + [navigator]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset, str(global_time),
                                      color='black', fontsize=10) for i in range(self.ped_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k-1][0].px, self.states[k][0].px),
                                               (self.states[k-1][0].py, self.states[k][0].py),
                                               color=navigator_color, ls='solid')
                    ped_directions = [plt.Line2D((self.states[k-1][1][i].px, self.states[k][1][i].px),
                                                 (self.states[k-1][1][i].py, self.states[k][1][i].py),
                                                 color=cmap(i), ls='solid')
                                      for i in range(self.ped_num)]
                    ax.add_artist(nav_direction)
                    for ped_direction in ped_directions:
                        ax.add_artist(ped_direction)
            # time = plt.text(-1, 6, 'Trajectories', fontsize=12)
            # ax.add_artist(time)
            plt.legend([navigator], ['Robot'], fontsize=20)
            # plt.title(self.navigator.policy.name)
            plt.show()
        elif mode == 'video':
            navigator_positions = [state[0].position for state in self.states]
            ped_positions = [[state[1][j].position for j in range(len(self.peds))] for state in self.states]

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            goal = plt.Circle((0, 4), 0.05, fill=True, color=goal_color)
            navigator = plt.Circle(navigator_positions[0], self.navigator.radius, fill=True, color=navigator_color)
            # visualize attention weights using the color saturation
            if self.attention_weights is not None:
                peds = [plt.Circle(ped_positions[0][i], self.peds[i].radius, fill=False,
                                   color=(self.attention_weights[0][i], 0, 0)) for i in range(len(self.peds))]
            else:
                peds = [plt.Circle(ped_positions[0][i], self.peds[i].radius, fill=False, color=str((i+1)/20))
                        for i in range(len(self.peds))]
            ped_annotations = [plt.text(peds[i].center[0]-x_offset, peds[i].center[1]-y_offset, str(i), color='black',
                                        fontsize=20) for i in range(len(self.peds))]
            time = plt.text(-1, 4.5, 'Time: {}'.format(0), fontsize=20)
            global_step = 0
            if self.attention_weights is not None:
                attention_scores = [plt.text(-4.5, 4.5 - 0.5 * i, 'Ped {}: {:.2f}'.format(i, self.attention_weights[0][i]),
                                             fontsize=20) for i in range(len(self.peds))]
            radius = self.navigator.radius
            if self.navigator.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].px + radius * np.cos(state[0].theta)),
                                (state[0].py, state[0].py + radius * np.sin(state[0].theta))) for state in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.ped_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i-1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.px + radius * np.cos(theta)),
                                           (agent_state.py, agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            orientation_segments = [plt.Line2D(*orientation[0], color=heading_color) for orientation in orientations]

            for segment in orientation_segments:
                ax.add_artist(segment)
            ax.add_artist(navigator)
            ax.add_artist(goal)
            ax.add_artist(time)
            for i, ped in enumerate(peds):
                ax.add_artist(ped)
                ax.add_artist(ped_annotations[i])
            plt.legend([navigator, goal], ['Robot', 'Goal'], fontsize=20)

            def update(frame_num):
                nonlocal global_step
                global_step = frame_num
                navigator.center = navigator_positions[frame_num]
                for i, ped in enumerate(peds):
                    ped.center = ped_positions[frame_num][i]
                    ped_annotations[i].set_position((ped.center[0]-x_offset, ped.center[1]-y_offset))
                    for agent_i, orientation in enumerate(orientations):
                        orientation_segments[agent_i].set_xdata(orientation[frame_num][0])
                        orientation_segments[agent_i].set_ydata(orientation[frame_num][1])
                    if self.attention_weights is not None:
                        ped.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('Ped {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.navigator.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.navigator.policy.speeds
                rotations = self.navigator.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                plt.subplot(projection="polar")
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                plt.colorbar(mesh, cax=cbaxes)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.navigator.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            if self.case_counter['test'] == -3:
                plot_value_heatmap()

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
