import numpy as np
import time
from geometry_msgs.msg import PoseStamped, Vector3
import configparser
import gym
import torch
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from matplotlib import animation

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.action import ActionXY


class SARLPolicyTest:

    def __init__(self, robot_data):
        self.robot_name = robot_data['name']
        self.robot_pref_speed = robot_data['pref_speed']
        self.robot_radius = robot_data['radius']
        self.robot_data = robot_data
        self.stop_moving_flag = False
        self.vel = Vector3()
        self.pos = Vector3()
        self.psi = 0
        self.goal = [robot_data['goal'][0], robot_data['goal'][1]]
        print "GOAL X: ", self.goal[0]
        print "GOAL Y: ", self.goal[1]
        self.ego_agent = [] # Plotting purposes
        self.other_agents = [] # Plotting purpose
        self.frame = 0
        self.data = []

        # External Agent(s) state
        self.other_agents_state = [ObservableState(5, 0, 0, 0, 0.3)]

        # what we use to send commands
        self.desired_action = ActionXY(0, 0)
        self.desired_position = PoseStamped()

        # SARL specific intializations
        model_dir = "crowd_nav/data/output/"
        phase = "test"
        model_weights = model_dir + "rl_model.pth"
        policy_config_file = model_dir + "policy.config"
        env_config_file = model_dir + "env.config"
        cuda = raw_input("Use cuda? (y/n): ")
        if cuda[0] == 'y' or cuda[0] == 'Y':
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print "============================"
                print "=== DEVICE INTIALIZATION ==="
                print "============================"
                print "Device: ", self.device
        else:
            self.device = torch.device("cpu")
            print "============================"
            print "=== DEVICE INTIALIZATION ==="
            print "============================"
            print "Device: ", self.device

        policy_config = configparser.RawConfigParser()
        policy_config.read(policy_config_file)

        self.policy = policy_factory["sarl"]()
        self.policy.configure(policy_config)

        if self.policy.trainable:
            print "SETTING MODEL WEIGHTS"
            self.policy.get_model().load_state_dict(torch.load(model_weights))

        env_config = configparser.RawConfigParser()
        env_config.read(env_config_file)

        self.env = gym.make('CrowdSim-v0')
        self.env.configure(env_config)

        self.robot = Robot(env_config, 'robot')
        self.robot.set_policy(self.policy)

        self.env.set_robot(self.robot)

        self.policy.set_phase(phase)
        self.policy.set_device(self.device)
        self.policy.set_env(self.env)
        # TODO: Need to set policy time step through config files.
        # NOTE: Manually set time step in Policy() base class (hacky workaround).

        self.robot.print_info()

        # control rate & nn update rate
        self.CONTROL_RATE = 0.001
        self.NN_RATE = 0.01
        self.GOAL_THRESH = 0.5

    def compute_action(self):
        # Set robot
        start_time = time.time()
        px = self.pos.x
        py = self.pos.y
        vx = self.vel.x
        vy = self.vel.y
        gx = self.goal[0]
        gy = self.goal[1]
        theta = self.psi
        radius = self.robot_radius
        v_pref = self.robot_pref_speed
        self.robot.set(px, py, gx, gy, vx, vy, theta, radius, v_pref)

        robot_dist_to_goal = np.linalg.norm(np.array([px, py]) - np.array([gx, gy]))

        if robot_dist_to_goal < self.GOAL_THRESH:
            self.stop_moving_flag = True
        else:
            self.stop_moving_flag = False

        # Communicate actions to the control node:
        print "\n================================================================\n"
        print "SARLPolicyNode.compute_action:"
        print "--->self.other_agents_state obj: ", self.other_agents_state
        print "--->self.other_agents_state pos: ", self.other_agents_state[0].position
        print "--->self.other_agents_state vel: ", self.other_agents_state[0].velocity
        self.desired_action = self.robot.act(self.other_agents_state)
        print "\n--->SARLPolicyNode.compute_action runtime: ", time.time() - start_time
        print "\n================================================================\n"

    def log_data(self):

        agents = []
        robot = [self.pos.x, self.pos.y, self.vel.x, self.vel.y, self.robot_radius]
        obstacles = [[other_agent.px, other_agent.py, other_agent.vx, other_agent.vy, other_agent.radius] for
                     other_agent in self.other_agents_state]
        agents.append(robot)
        for obs in obstacles:
            agents.append(obs)
        self.data.append((self.frame, agents))

    def take_action(self):
        if self.stop_moving_flag:
            self.frame += 1
            self.log_data()
            return
        else:
            self.vel.x = self.desired_action.vx
            self.vel.y = self.desired_action.vy
            self.pos.x = self.pos.x + self.vel.x * self.policy.time_step
            self.pos.y = self.pos.y + self.vel.y * self.policy.time_step
            self.frame += 1
            self.log_data()
            return


def run():
    print '\n===============================================\n'
    print 'Running test version of SARL without ROS nodes.'
    print '\n===============================================\n'

    while True:
        try:
            goal = tuple(input('Robot start position is (0,0). Enter a goal: '))
            break
        except TypeError:
            print "\nPlease enter goal as a tuple.\n"
    print goal, ", ", goal[0], ", ", goal[1]
    robot_data = {'goal': goal, 'radius': 0.3, 'pref_speed': 0.5, 'name': 'balabot'}
    sarl = SARLPolicyTest(robot_data)
    steps = 0

    while steps < 100:
        sarl.compute_action()
        sarl.take_action()
        steps += 1

    return

def init():
    return []


def animate(frame, sarl):
    frame_data = sarl.data[frame]
    agents = frame_data[1]
    markers = []
    for i, agent in enumerate(agents):
        agent_x = agent[0]
        agent_y = agent[1]
        agent_rad = agent[4]
        if i == 0:
            patch = ax.add_patch(plt.Circle((agent_x, agent_y), agent_rad, color='b'))
        else:
            patch = ax.add_patch(plt.Circle((agent_x, agent_y), agent_rad, color='r'))
        markers.append(patch)
    return markers



if __name__ == '__main__':
    print '\n===============================================\n'
    print 'Running test version of SARL without ROS nodes.'
    print '\n===============================================\n'

    while True:
        try:
            goal = tuple(input('Robot start position is (0,0). Enter a goal: '))
            break
        except TypeError:
            print "\nPlease enter goal as a tuple.\n"
    print goal, ", ", goal[0], ", ", goal[1]
    robot_data = {'goal': goal, 'radius': 0.3, 'pref_speed': 0.5, 'name': 'balabot'}
    sarl = SARLPolicyTest(robot_data)
    steps = 0

    while steps < 150:
        sarl.compute_action()
        sarl.take_action()
        steps += 1

    fig = plt.figure(figsize=(15,15))
    plt.axis('equal')
    plt.grid()
    ax = fig.add_subplot(111)
    ax.set_aspect('auto')
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=100, fargs=(sarl,), blit=True)
    plt.show()
