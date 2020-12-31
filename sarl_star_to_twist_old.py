import rospy
import numpy as np
import time
from tf import transformations
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from gazebo_msgs.msg import ModelState, ModelStates
import configparser
import gym
import torch

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.action import ActionXY


class SARLPolicy:

    def __init__(self, robot_data):

        self.robot_radius = robot_data['radius']
        self.robot_pref_speed = robot_data['pref_speed']  # TODO: Make this a dynamic variable
        self.stop_moving_flag = False
        self.state = ModelState()
        self.STATE_SET_FLAG = False
        self.goal = PoseStamped()
        self.GOAL_RECEIVED_FLAG = False
        self.GOAL_THRESH = 0.5

        # External Agent(s) state
        self.other_agents_state = [ObservableState(float("inf"), float("inf"), 0, 0, 0.3)]
        self.OBS_RECEIVED_FLAG = False

        # what we use to send commands
        self.desired_action = ActionXY(0, 0)

    def compute_action(self):

        start_time = time.time()

        # Set robot
        px, py = self.state.pose.position.x, self.state.pose.position.y
        vx, vy = self.state.twist.linear.x, self.state.twist.linear.y
        gx, gy = self.goal.pose.position.x, self.goal.pose.position.y

        q = self.state.pose.orientation
        _, _, yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        theta = yaw

        radius, v_pref = self.robot_radius, self.robot_pref_speed

        robot.set(px, py, gx, gy, vx, vy, theta, radius, v_pref)

        robot_dist_to_goal = np.linalg.norm(np.array([px, py]) - np.array([gx, gy]))

        if robot_dist_to_goal < self.GOAL_THRESH:
            self.stop_moving_flag = True
            return Twist()
        else:
            self.stop_moving_flag = False

        # Communicate actions to the control node:A
        print "\n================================================================\n"
        print "SARLPolicyNode.compute_action:"
        print "--->self.other_agents_state pos: ", self.other_agents_state[0].position
        print "--->self.other_agents_state vel: ", self.other_agents_state[0].velocity
        print "--->self.pos: ", self.state.pose.position.x
        print "--->self.vel: ", np.linalg.norm([self.state.twist.linear.x, self.state.twist.linear.y])

        self.desired_action = robot.act(self.other_agents_state)
        twist = Twist()
        twist.linear.x = self.desired_action.vx
        twist.linear.y = self.desired_action.vy
        print "\n--->SARLPolicyNode.compute_action runtime: ", time.time() - start_time
        print "\n================================================================\n"
        return twist

    def update_state(self, robot_state):
        self.state = robot_state
        self.STATE_SET_FLAG = True

    def update_dynamic_goal(self, msg):
        self.GOAL_RECEIVED_FLAG = True
        new_goal = PoseStamped()
        new_goal.pose.position.x = msg.pose.position.x
        new_goal.pose.position.y = msg.pose.position.y
        self.goal = new_goal

    def set_other_agents(self, humans):
        self.OBS_RECEIVED_FLAG = True
        self.other_agents_state = humans


def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def cb_state(msg):
    global state
    global STATE_SET
    state.pose.position = msg.pose.position
    state.twist.linear = msg.twist.linear
    STATE_SET = True


def cb_other_agents(msg):
    # Create list of HUMANS
    global other_agents
    global OTHER_AGENTS_SET
    other_agents = []
    num_agents = len(msg.name)
    for i in range(num_agents):
        radius = 0.3  # Spheres in gazebo
        x = msg.pose[i].position.x
        y = msg.pose[i].position.y
        vx = msg.twist[i].linear.x
        vy = msg.twist[i].linear.y
        other_agents.append(ObservableState(x, y, vx, vy, radius))
    OTHER_AGENTS_SET = True


def cb_dynamic_goal(msg):
    global goal
    global GOAL_SET
    goal.pose.position.x = msg.pose.position.x
    goal.pose.position.y = msg.pose.position.y
    GOAL_SET = True



def cb_real_other_agent(msg):
    global other_agents
    global OTHER_AGENTS_SET
    other_agents = []
    num_agents = len(msg.name)
    for i in range(num_agents):
        x = msg.pose[i].position.x
        y = msg.pose[i].position.y
        vx = 0
        vy = 0
        other_agents.append(ObservableState(x,y,vx,vy,0.3))


def cb_real_pose(msg):
    global state
    global STATE_SET
    state.pose.position = msg.pose.position
    STATE_SET = True


def initialize_robot():
    model_dir = "crowd_nav/data/output/"
    phase = "test"
    model_weights = model_dir + "rl_model.pth"
    policy_config_file = model_dir + "policy.config"
    env_config_file = model_dir + "env.config"
    cuda = raw_input("Set device as Cuda? (y/n)")
    if torch.cuda.is_available() and cuda == 'y':
        device = torch.device("cuda:0")
        print "================================"
        print "=== Device: ", device, "==="
        print "================================"
    else:
        device = torch.device("cpu")
        print "===================="
        print "=== Device: ", device, "==="
        print "===================="

    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)

    policy = policy_factory["sarl"]()
    policy.configure(policy_config)

    if policy.trainable:
        print "SETTING MODEL WEIGHTS"
        policy.get_model().load_state_dict(torch.load(model_weights))

    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)

    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)

    env.set_robot(robot)

    policy.set_phase(phase)
    policy.set_device(device)
    policy.set_env(env)
    # TODO: NEED TO SET POLICY TIME_STEP

    return robot


if __name__ == '__main__':
    print('About to run SARL')

    # SARL specific intializations
    robot = initialize_robot()
    
    try:
        state = ModelState()
        STATE_SET = False
        goal = PoseStamped()
        GOAL_SET = False
        other_agents = []
        OTHER_AGENTS_SET = False
        robot_data = {'goal': None, 'radius': 0.3, 'pref_speed': 0.8, 'name': 'balabot'}
        sarl_policy_node = SARLPolicy(robot_data)
        rospy.init_node(robot_data['name'], anonymous=False)
        rate = rospy.Rate(10)
        node_name = rospy.get_name()
        scenario = input("Running in real or gazebo?\n (1 for real, 2 for gazebo):")
        if scenario == 2:
            sub_other_agents = rospy.Subscriber('/sphere_states', ModelStates, cb_other_agents)
            sub_pose = rospy.Subscriber('/sim_pathbot_state', ModelState, cb_state)
            sub_goal = rospy.Subscriber('/sim_pathbot_goal', PoseStamped, cb_dynamic_goal)
            control_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        else:
            sub_other_agent1 = rospy.Subscriber('/vrpn_client_node/agent1/pose', PoseStamped, self.cb_real_other_agent)
            sub_goal = rospy.Subscriber('vrpn_client_node/agent3/pose', PoseStamped, self.cb_dynamic_goal)
            sub_pose = rospy.Subscriber('/pathbot/pose', PoseStamped, self.cb_real_pose)
        while not rospy.is_shutdown():
            if STATE_SET and GOAL_SET and OTHER_AGENTS_SET:
                sarl_policy_node.update_state(state)
                sarl_policy_node.update_dynamic_goal(goal)
                sarl_policy_node.set_other_agents(other_agents)
                control_pub.publish(sarl_policy_node.compute_action())
            rate.sleep()
    except rospy.ROSInterruptException, e:
        raise e
