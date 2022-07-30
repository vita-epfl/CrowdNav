from curses import nonl
import logging
from crowd_nav.policy.gat4sn import GAT4SN
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by an unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.obs = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_dist_front = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

        # limited FOV
        self.robot_fov = None
        self.human_fov = None
        self.uncertainty_growth = None
        # todo: i didnt like this idea of dummy human & robot.
        #self.dummy_human = None
        #self.dummy_robot = None

        # curriculum learning
        self.largest_obst_ratio = 0.0
        self.cl_radius_start = None
        self.cl_radius_max = None
        self.radius_increment = None


    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_dist_front = config.getfloat('reward', 'discomfort_dist_front') #1
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.out_boundary_penalty = config.getfloat('reward', 'out_boundary_penalty')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
            ## Parameter for static obstacles
            self.min_obst_offset = config.getfloat('sim', 'min_obst_offset')
            self.static_obstacle_num = config.getint('sim', 'static_obstacle_num')
            # the following two will be overwritten in curriculum learning mode
            self.obstacle_max_radius = config.getfloat('sim', 'obstacle_max_radius')
            self.obstacle_min_radius = config.getfloat('sim', 'obstacle_min_radius')

            self.boundary = config.getfloat('sim', 'boundary')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))
        logging.info("Number of static obstacles: {}".format(self.static_obstacle_num))
        logging.info("Static obstacles diameter range: {} - {}".format(2 * self.obstacle_min_radius, 2 * self.obstacle_max_radius))

        #Fov Config
        self.robot_fov = np.pi * config.getfloat('robot' , 'FOV')
        self.human_fov = np.pi * config.getfloat('humans', 'FOV')
        self.uncertainty_growth = config.get('sim', 'uncertainty_growth')

        # # set dummy human and dummy robot
        # # dummy humans, used if any human is not in view of other agents
        # self.dummy_human = Human(self.config, 'humans')
        # # if a human is not in view, set its state to (px = 100, py = 100, vx = 0, vy = 0, theta = 0, radius = 0)
        # self.dummy_human.set(7, 7, 7, 7, 0, 0, 0) # (7, 7, 7, 7, 0, 0, 0)
        # self.dummy_human.time_step = config.getfloat('env', 'time_step')
        #
        # self.dummy_robot = Robot(self.config, 'robot')
        # self.dummy_robot.set(7, 7, 7, 7, 0, 0, 0)
        # self.dummy_robot.time_step = config.getfloat('env', 'time_step')
        # self.dummy_robot.kinematics = 'holonomic'
        # self.dummy_robot.policy = ORCA(config)

    def configure_cl(self,train_config):
        # curriculum learning
        mode = train_config.get('curriculum', 'mode') # 'increasing_obst_num','single obstacle in the middle
        self.cl_radius_start = train_config.getfloat('curriculum','radius_start')
        self.obstacle_max_radius = self.obstacle_min_radius = self.cl_radius_start
        self.cl_radius_max = train_config.getfloat('curriculum','radius_max')
        self.radius_increment = train_config.getfloat('curriculum','radius_increment')
        self.largest_obst_ratio = train_config.getfloat('curriculum','largest_obst_ratio')
        level_up_mode = train_config.get('curriculum', 'level_up_mode')
        success_rate_milestone = train_config.getfloat('curriculum','success_rate_milestone')
        self.p_handcrafted = train_config.getfloat('curriculum','p_handcrafted')
        p_hard_deck = train_config.getfloat('curriculum','p_hard_deck')
        hard_deck_cap = train_config.getint('curriculum','hard_deck_cap')


    def set_robot(self, robot):
        self.robot = robot
        if self.robot.sensor == 'RGB':
            logging.info('robot FOV %f', self.robot_fov)
            logging.info('humans FOV %f', self.human_fov)
            logging.info('uncertainty growth mode: %s', self.uncertainty_growth)

    def set_max_obst_r(self,r_max):
        self.obstacle_max_radius = r_max
        if r_max > self.cl_radius_max:
            self.obstacle_max_radius = self.cl_radius_max
            return False
        else:
            self.obstacle_max_radius = r_max
            return True

    def increase_cl_level(self):
        print('Level increase prompted in env: Setting max_obstacle_radius from {} to {}'.format(
            self.obstacle_max_radius,
            self.obstacle_max_radius+self.radius_increment
        ))
        success = self.set_max_obst_r(self.obstacle_max_radius+self.radius_increment)
        if success:
            print("Success: Obstacle radius increased successfully.")
        else:
            print("Fail: Max obstacle radius is reached. Setting max_radius to max radius given in curriculum learning config.")
            return success

    def generate_random_obstacles(self, obs_num,phase):
        width = self.square_width
        height = self.square_width
        if phase == 'train':
            radius_offset = self.obstacle_max_radius - self.obstacle_min_radius
            max_radius = self.obstacle_max_radius
            min_radius = self.obstacle_min_radius
        # Validation or test is on hardest level
        else:
            radius_offset = self.cl_radius_max - self.cl_radius_start
            max_radius = self.cl_radius_max
            min_radius = self.cl_radius_start
        self.obs = []

        large_obst_num = np.ceil(obs_num * self.largest_obst_ratio).astype(np.int64).item()
        other_obst_num = obs_num - large_obst_num

        for i in range(large_obst_num):
            human = Human(self.config, 'humans') ## we model the static obstacles as static humans
            while True:
                px = (np.random.random() - 0.5) * width
                py = (np.random.random() - 0.5) * height
                r = max_radius
                collide = False
                for agent in [self.robot] + self.obs:
                    if norm((px - agent.px, py - agent.py)) < r + agent.radius + self.discomfort_dist + self.min_obst_offset or norm((px - self.robot_gx, py - self.robot_gy)) < r + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, px, py, 0, 0, 0, radius=r)
            # print("Generate obstacle!")
            self.obs.append(human)

        for i in range(other_obst_num):
            human = Human(self.config, 'humans') ## we model the static obstacles as static humans
            while True:
                px = (np.random.random() - 0.5) * width
                py = (np.random.random() - 0.5) * height
                r = (np.random.random()) * radius_offset + min_radius
                collide = False
                for agent in [self.robot] + self.obs:
                    if norm((px - agent.px, py - agent.py)) < r + agent.radius + self.discomfort_dist + self.min_obst_offset or norm((px - self.robot_gx, py - self.robot_gy)) < r + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, px, py, 0, 0, 0, radius=r)
            # print("Generate obstacle!")
            self.obs.append(human)
    # This function was written to generate handcrafted hard cases for robot to learn faster,
    # but for this reason we set humans very far from the robot. This
    # breaks the attention mechanism and network generates nan values. We are not using it anymore
    def generate_obstacle_in_center(self, obs_num,phase):
        width = self.square_width
        height = self.square_width
        self.obs = []
        # generate single obstacle in center
        human = Human(self.config, 'humans') ## we model the static obstacles as static humans
        px = 0.0
        py = 0.0
        if phase == 'train':
            r = self.obstacle_max_radius
        # Validation or test is on hardest level
        else:
            r = self.cl_radius_max
        human.set(px, py, px, py, 0, 0, 0, radius=r)
        # print("Generate obstacle!")
        self.obs.append(human)
        # put rest of the obstacles outside the simulation
        for i in range(obs_num-1):
            human = Human(self.config, 'humans') ## we model the static obstacles as static humans
            # hack: we can't input varius number of input, so we put other obstacles very far
            px = 99 * width
            py = 99 * height
            r = 0.3
            human.set(px, py, px, py, 0, 0, 0, radius=r)
            # print("Generate obstacle!")
            self.obs.append(human)

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        elif rule == 'test': ## Only test for generating static obstacles
            self.generate_random_obstacles(human_num)
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            # for agent in [self.robot] + self.humans:
            for agent in [self.robot] + self.humans + self.obs:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            # for agent in [self.robot] + self.humans:
            for agent in [self.robot] + self.humans + self.obs:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    # todo: add noise according to env.config to observation
    # def apply_noise(self, ob):
    #     if isinstance(ob[0], ObservableState):
    #         for i in range(len(ob)):
    #             if self.noise_type == 'uniform':
    #                 noise = np.random.uniform(-self.noise_magnitude, self.noise_magnitude, 5)
    #             elif self.noise_type == 'gaussian':
    #                 noise = np.random.normal(size=5)
    #             else:
    #                 print('noise type not defined')
    #             ob[i].px = ob[i].px + noise[0]
    #             ob[i].py = ob[i].px + noise[1]
    #             ob[i].vx = ob[i].px + noise[2]
    #             ob[i].vy = ob[i].px + noise[3]
    #             ob[i].radius = ob[i].px + noise[4]
    #         return ob
    #     else:
    #         if self.noise_type == 'uniform':
    #             noise = np.random.uniform(-self.noise_magnitude, self.noise_magnitude, len(ob))
    #         elif self.noise_type == 'gaussian':
    #             noise = np.random.normal(size = len(ob))
    #         else:
    #             print('noise type not defined')
    #             noise = [0] * len(ob)
    #
    #         return ob + noise

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5) # kagan: i guess 10,10 is the box size of the environment and 5,5 is where the origin is
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def generate_agent_goal(self, goal_range = 8, perturb = False, perturb_range = 1):
        if perturb:
            px = (np.random.random() - 0.5) * perturb_range
            py = (np.random.random() - 0.5) * perturb_range
        else:
            px = 0
            py = 0
        angle = np.random.random() * 2 * np.pi
        gx = goal_range * np.cos(angle) + px
        gy = goal_range * np.sin(angle) + py
        return gx, gy

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            # we should make the goal position more diverse
            ## The random seed should also be added here, otherwise the
            ## generated environment would be totally different
            np.random.seed(counter_offset[phase] + self.case_counter[phase])
            while True:
                self.robot_gx, self.robot_gy = self.generate_agent_goal(goal_range = self.square_width / 2)
                self.robot_px, self.robot_py = self.generate_agent_goal(perturb = True, perturb_range = (self.boundary - self.square_width) / 2, goal_range = self.square_width / 2)

                if np.abs(self.robot_gx - self.robot_px) > self.boundary / 2 or np.abs(self.robot_gy - self.robot_py) > self.boundary / 2:
                    break

            self.robot.set(self.robot_px, self.robot_py, self.robot_gx, self.robot_gy, 0, 0, np.pi / 2)

            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                ## Geneate static obstacles first
                self.generate_random_obstacles(self.static_obstacle_num, phase)
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                elif self.case_counter[phase] == -2:
                    # for testing to generate static obstacle
                    self.generate_random_human_position(human_num=self.human_num, rule='square_crossing')
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        self.observable_states = list()
        if self.robot.sensor == 'coordinates':
            ## Let the static obstacles also generate their states
            # ob = [human.get_observable_state() for human in self.humans]
            ob = [human.get_observable_state() for human in self.humans]
            temp = [obstacle.get_observable_state() for obstacle in self.obs]
            ob += temp
        elif self.robot.sensor == 'RGB':
            humans_in_view, num_humans_in_view, seen_human_ids, unseen_human_ids  = self.get_num_human_in_fov()
            for human in humans_in_view:
                human.increment_uncertainty('reset')
            for id in unseen_human_ids:
                self.humans[id].increment_uncertainty(self.uncertainty_growth)
            ob = [human.get_observable_state() for human in self.humans]
            temp = [obstacle.get_observable_state() for obstacle in self.obs]
            ob += temp

        return ob

    # Caculate whether agent2 is in agent1's FOV
    # Not the same as whether agent1 is in agent2's FOV!!!!
    # arguments:
    # state1, state2: can be agent instance OR state instance
    # robot1: is True if state1 is robot, else is False
    # return value:
    # return True if state2 is visible to state1, else return False
    def detect_visible(self, state1, state2, robot1 = False, custom_fov=None):
        #todo: add obstacle checking
        if self.robot.kinematics == 'holonomic':
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov
            else:
                fov = self.human_fov

        if np.abs(offset) <= fov / 2:
            return True
        else:
            return False

    # for robot:
    # return only visible humans to robot and number of visible humans and visible humans' ids (0 to 4)
    def get_num_human_in_fov(self):
        seen_human_ids = []
        unseen_human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        for i in range(len(self.humans)):
            visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view += 1
                seen_human_ids.append(i)
            else:
                unseen_human_ids.append(i)
        # for i in range(self.human_num):
        #     visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
        #     if visible:
        #         humans_in_view.append(self.humans[i])
        #         num_humans_in_view += 1
        #         human_ids.append(True)
        #     else:
        #         human_ids.append(False)

        return humans_in_view, num_humans_in_view, seen_human_ids, unseen_human_ids

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def generate_valid_goal(self, gx, gy, r):
        while True:
            collide = False
            for agent in [self.robot] + self.humans + self.obs:
                if norm((gx - agent.gx, gy - agent.gy)) < r + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        return gx, gy

    def human_reset_goal(self, human):
        # eps = 1e-6
        px, py = human.get_position()
        vx, vy = human.get_velocity()
        gx, gy = human.get_goal_position()
        # still = False
        # if vx < eps and vy < eps:
        #     still = True

        if human.reached_destination():
            gx, gy = self.generate_agent_goal(goal_range = self.square_width / 2)
            human.set(px, py, -gx, -gy, 0, 0, 0)
        # elif still:
        #     human.set(px, py, -human.gx, -human.gy, 0, 0, 0)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            temp = [obstacle.get_observable_state() for obstacle in self.obs]
            ob += temp
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            # kagan: discomfort distance is added later as a penalty.
            # adding it above would set it as collision and stop the episode.
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # for static obstacle collision detection
        for i, obstacle in enumerate(self.obs):
            # print("obstacle {0:d}: x: {1:f} y: {2:f}".format(i, obstacle.px, obstacle.py))
            px = obstacle.px - self.robot.px
            py = obstacle.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = obstacle.vx - action.vx
                vy = obstacle.vy - action.vy
            else:
                vx = obstacle.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = obstacle.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - obstacle.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # print("Collision!")
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        ## check if the robot run out of the boundary
        robot_x, robot_y = self.robot.get_position()
        out = False

        closest_dist_to_bd_x = self.boundary / 2 - (np.abs(end_position[0]) + self.robot.radius)
        closest_dist_to_bd_y = self.boundary / 2 - (np.abs(end_position[1]) + self.robot.radius)

        if closest_dist_to_bd_x < 0:
            out = True
        else:
            if closest_dist_to_bd_x < dmin:
                dmin = closest_dist_to_bd_x
        if closest_dist_to_bd_y < 0:
            out = True
        else:
            if closest_dist_to_bd_y < dmin:
                dmin = closest_dist_to_bd_y

        # if np.abs(end_position[0]) + self.robot.radius > self.boundary / 2 or np.abs(end_position[1]) + self.robot.radius > self.boundary / 2:
        #     out = True

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif out:
            reward = self.out_boundary_penalty
            done = True
            info = Collision()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS # kagan: nice! time step weights penalty
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        if update: # kagan: update == false if doing one_step_look_ahead
            # store state, action value and attention weights
            # env_obs = [human.get_full_state() for human in self.humans]
            # temp = [obstacle.get_full_state() for obstacle in self.obs]
            # env_obs += temp
            for human in self.humans:
                self.human_reset_goal(human) ## If human already reached its goal state, reset its goal

            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans], [obstacle.get_full_state() for obstacle in self.obs]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            # store observable states
            self.observable_states.append([human.get_observable_state() for human in self.humans])

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
                temp = [obstacle.get_observable_state() for obstacle in self.obs]
                ob += temp
            elif self.robot.sensor == 'RGB':
                humans_in_view, num_humans_in_view, seen_human_ids, unseen_human_ids  = self.get_num_human_in_fov()
                for human in humans_in_view:
                    human.increment_uncertainty('reset')
                for id in unseen_human_ids:
                    self.humans[id].increment_uncertainty('logarithmic')
                ob = [human.get_observable_state() for human in self.humans]
                temp = [obstacle.get_observable_state() for obstacle in self.obs]
                ob += temp


        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
                # todo: check it with this version
                temp = [obstacle.get_observable_state() for obstacle in self.obs]
                ob += temp
            elif self.robot.sensor == 'RGB':
                humans_in_view, num_humans_in_view, seen_human_ids, unseen_human_ids  = self.get_num_human_in_fov()
                for human in humans_in_view:
                    human.increment_uncertainty('reset')
                for id in unseen_human_ids:
                    self.humans[id].increment_uncertainty('logarithmic')
                ob = [human.get_observable_state() for human in self.humans]
                temp = [obstacle.get_observable_state() for obstacle in self.obs]
                ob += temp

        return ob, reward, done, info

    def render(self, mode='human', output_file=None, debug = False):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-(self.boundary / 2 + 2), (self.boundary / 2 + 2))
            ax.set_ylim(-(self.boundary / 2 + 2), (self.boundary / 2 + 2))
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            # goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            goal = mlines.Line2D([self.robot_gx], [self.robot_gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            boundary = plt.Rectangle((-self.boundary / 2, -self.boundary / 2), self.boundary, self.boundary,
             edgecolor = 'black',
             fill=False,
             lw=5)
            ax.add_artist(robot)
            ax.add_artist(goal)
            ax.add_patch(boundary)
            plt.legend([robot, goal, boundary], ['Robot', 'Goal', 'Boundary'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add human observable positions
            if self.robot.sensor == 'RGB':
                human_observations = []
                observed_human_numbers = []
                human_uncertainties = []
                if debug:
                    observed_vels = []
                for i in range(len(self.observable_states[0])):
                    human_observations.append(plt.Circle((self.observable_states[0][i].px,
                                                        self.observable_states[0][i].py),
                                                       radius=self.observable_states[0][i].radius,
                                                       alpha = 0.3,
                                                         color='blue'))
                    observed_human_numbers.append(plt.text(self.observable_states[0][i].px - x_offset,
                                                            self.observable_states[0][i].py - y_offset,
                                                           str(i),
                                                           color = 'blue',
                                                           fontsize=12))
                    human_uncertainties.append(plt.text(-5.5,
                                                        -4 - 0.4 * i,
                                                        'Uncertainty {}: {:.2f}'.format(i, self.observable_states[0][i].uncertainty),
                                                        fontsize=12))
                    if debug:
                        observed_vels.append(plt.text(  3,
                                                        -4 - 0.4 * i,
                                                        'Vel. {}: {:.1f},{:.1f}'.format(i,
                                                                                        self.observable_states[0][i].vx,
                                                                                        self.observable_states[0][i].vy),
                                                        fontsize=12))

                for obs in human_observations:
                    ax.add_artist(obs)
                for num in observed_human_numbers:
                    ax.add_artist(num)

            # add obs and their numbers
            obstacle_positions = [[state[2][j].position for j in range(len(self.obs))] for state in self.states]
            obstacles = [plt.Circle(obstacle_positions[0][i], self.obs[i].radius, fill=True, color='black')
                      for i in range(len(self.obs))]
            obstacle_numbers = [plt.text(obstacles[i].center[0] - x_offset, obstacles[i].center[1] - y_offset, str(i),
                                      color='white', fontsize=12) for i in range(len(self.obs))]
            for i, ob in enumerate(obstacles):
                ax.add_artist(ob)
                ax.add_artist(obstacle_numbers[i])
            # add time annotation
            time = plt.text(-1, 8.5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)


            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(self.boundary / 2 + 0.5, 5 - 0.6 * i, 'Human {}: {:.3f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=12) for i in range(len(self.humans))]
                attention_obstacle_scores = [
                    plt.text(self.boundary / 2 + 0.5, 5 - 0.6 * (i + len(self.humans)), 'Obstacle {}: {:.3f}'.format(i + 1, self.attention_weights[0][i + len(self.humans)]),
                             fontsize=12) for i in range(len(self.obs))]

            if self.robot.policy.name == 'GAT4SN':
                max_edge_width = 30
                alpha = 0.5
                edge_color = 'red'
                edges_to_humans = [plt.Line2D([robot_positions[0][0], human_positions[0][i][0]], [robot_positions[0][1], human_positions[0][i][1]], linestyle = '--', linewidth = self.attention_weights[0][i] * max_edge_width, color = edge_color, alpha = alpha) for i in range (len(self.humans))]
                edges_to_obstacles = [plt.Line2D([robot_positions[0][0], obstacle_positions[0][i][0]], [robot_positions[0][1], obstacle_positions[0][i][1]], linestyle = '--', linewidth = self.attention_weights[0][i + len(humans)] * max_edge_width, color = edge_color, alpha = alpha) for i in range (len(self.obs))]
                edges = edges_to_humans + edges_to_obstacles
                for i, edge in enumerate(edges):
                    ax.add_artist(edge)

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0


            ## SHOW FOV
            def calcFOVLineEndPoint(ang, point, extendFactor):
                # choose the extendFactor big enough
                # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
                FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                       [np.sin(ang), np.cos(ang), 0],
                                       [0, 0, 1]])
                point.extend([1])
                # apply rotation matrix
                newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
                # increase the distance between the line start point and the end point
                newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
                return newPoint

            if self.robot.sensor == 'RGB':
                FOVAng = self.robot_fov / 2
                fov_line_1_x_data = []
                fov_line_1_y_data = []
                fov_line_2_x_data = []
                fov_line_2_y_data = []
                for i in range(len(robot_positions)):
                    startPointX = orientations[0][i][0][0]
                    startPointY = orientations[0][i][0][1]
                    endPointX   = orientations[0][i][1][0]
                    endPointY   = orientations[0][i][1][1]
                    # endPointX = startPointX + radius * np.cos(orientation[i])
                    # endPointY = startPointY + radius * np.sin(orientation[i])
                    end_points1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
                    end_points2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
                    fov_line_1_x_data.append([startPointX,end_points1[0]])
                    fov_line_1_y_data.append([startPointY,end_points1[1]])
                    fov_line_2_x_data.append([startPointX,end_points2[0]])
                    fov_line_2_y_data.append([startPointY,end_points2[1]])

                FOVLine1 = plt.Line2D(fov_line_1_x_data[0], fov_line_1_y_data[0], linestyle='--')
                FOVLine2 = plt.Line2D(fov_line_2_x_data[0], fov_line_2_y_data[0], linestyle='--')
                ax.add_artist(FOVLine1)
                ax.add_artist(FOVLine2)

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows

                global_step = frame_num
                robot.center = robot_positions[frame_num]
                if self.robot.sensor == 'RGB':
                    FOVLine1.set_data(fov_line_1_x_data[frame_num],fov_line_1_y_data[frame_num])
                    FOVLine2.set_data(fov_line_2_x_data[frame_num],fov_line_2_y_data[frame_num])
                    for i, obs in enumerate(human_observations):
                        obs.center = (self.observable_states[frame_num][i].px,self.observable_states[frame_num][i].py)
                        obs.radius = self.observable_states[frame_num][i].radius
                        observed_human_numbers[i].set_position((self.observable_states[frame_num][i].px - x_offset,
                                                                self.observable_states[frame_num][i].py - y_offset))
                        human_uncertainties[i].set_text('Uncertainty {}: {:.2f}'.format(i, self.observable_states[frame_num][i].uncertainty))
                        if debug:
                            observed_vels[i].set_text('Vel. {}: {:.1f},{:.1f}'.format(i,
                                                                                      self.observable_states[frame_num][i].vx,
                                                                                      self.observable_states[frame_num][i].vy))
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.3f}'.format(i, self.attention_weights[frame_num][i]))

                for i, ob in enumerate(obstacles):
                    if self.attention_weights is not None:
                        ob.set_color(str(self.attention_weights[frame_num][i + len(self.humans)]))
                        attention_obstacle_scores[i].set_text('obstacle {}: {:.3f}'.format(i, self.attention_weights[frame_num][i + len(self.humans)]))

                if self.robot.policy.name == 'GAT4SN':
                    nonlocal edges
                    nonlocal edge_color
                    nonlocal alpha
                    nonlocal max_edge_width
                    for edge in edges:
                        edge.remove()

                    alpha = (self.attention_weights[frame_num] - np.min(self.attention_weights[frame_num])) / (np.max(self.attention_weights[frame_num]) - np.min(self.attention_weights[frame_num]))
                    edges_to_humans = [plt.Line2D([robot_positions[frame_num][0], human_positions[frame_num][i][0]], [robot_positions[frame_num][1], human_positions[frame_num][i][1]], linestyle = '--', linewidth = self.attention_weights[frame_num][i] * max_edge_width, color = edge_color, alpha = alpha[i]) for i in range (len(self.humans))]
                    edges_to_obstacles = [plt.Line2D([robot_positions[frame_num][0], obstacle_positions[frame_num][i][0]], [robot_positions[frame_num][1], obstacle_positions[frame_num][i][1]], linestyle = '--', linewidth = self.attention_weights[frame_num][i + len(humans)] * max_edge_width, color = edge_color, alpha = alpha[i + len(humans)]) for i in range (len(self.obs))]
                    edges = edges_to_humans + edges_to_obstacles

                    for i, edge in enumerate(edges):
                        ax.add_artist(edge)

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = np.append(self.robot.policy.rotations, np.pi * 2)

                angle_offset = (rotations[1] - rotations[0]) / 2
                rotations = rotations - angle_offset
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (len(rotations) - 1, len(speeds) - 1))

                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True


            if output_file is not None:
                # ffmpeg_writer = animation.writers['ffmpeg']
                # writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                # anim.save(output_file, writer=writer)
                writergif = animation.PillowWriter(fps = 8)
                anim.save(output_file, writer=writergif)
                plt.show()
            else:
                plt.show()
        else:
            raise NotImplementedError
