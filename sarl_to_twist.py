import rospy
import numpy as np
import time
from collections import deque
from tf import transformations
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import Bool, Float32
import configparser
import gym
import torch

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.action import ActionXY


class SARLPolicyNode:

    def __init__(self, robot_data):

        # Shared Initializations between SARL and CADRL
        rospy.init_node('PolicyNode', anonymous=False)
        self.node_name = rospy.get_name()
        self.robot_name = robot_data['name']
        self.robot_data = robot_data
        self.robot_pref_speed = 0.3
        self.stop_moving_flag = False
        self.vel = Vector3()
        self.pos = Vector3()
        self.psi = 0
        self.goal = PoseStamped()

        # External Agent(s) state
        self.other_agents_state = [ObservableState(float("inf"), float("inf"), 0, 0, 0.3)]

        # what we use to send commands
        self.desired_action = ActionXY(0, 0)
        self.desired_position = PoseStamped()

        # SARL specific intializations
        model_dir = "crowd_nav/data/output/"
        phase = "test"
        model_weights = model_dir + "rl_model.pth"
        policy_config_file = model_dir + "policy.config"
        env_config_file = model_dir + "env.config"
        cuda = input("Set device as Cuda? (y/n)")
        if torch.cuda.is_available() and cuda == 'y':
            self.device = torch.device("cuda:0")
            print "================================"
            print "=== Device: ", self.device, "==="
            print "================================"
        else:
            self.device = torch.device("cpu")
            print "================================"
            print "=== Device: ", self.device, "==="
            print "================================"

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
        # TODO: NEED TO SET POLICY TIME_STEP

        self.robot.print_info()


        # Publisher (Gazebo and Real)
        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Gazebo:
        #   Other agents in Gazebo:
        self.sub_other_agents = rospy.Subscriber('/sphere_states', ModelStates, self.cb_other_agents)
        # self.sub_other_agents = rospy.Subscriber('/vrpn_client_node/agent2/pose', PoseStamped, self.cb_other_agents)

        #   Pathbot pose in Gazebo:
        self.sub_pose = rospy.Subscriber('/sim_pathbot_state', ModelState, self.cb_pose)
        # self.sub_pose = rospy.Subscriber('/vrpn_client_node/pathbot/pose', PoseStamped, self.cb_pose)

        #   Pathbot goal in Gazebo:
        self.sub_goal = rospy.Subscriber('/sim_pathbot_goal', PoseStamped, self.cb_dynamic_goal)
        # self.sub_goal = rospy.Subscriber('/vrpn_client_node/agent1/pose', PoseStamped, self.cb_dynamic_goal)
        # Moving flag publisher in Gazebo:
        self.flag_pub = rospy.Publisher('sim_pathbot_move_flag', Bool, queue_size=10)

        # control rate & nn update rate
        self.CONTROL_RATE = 0.001
        self.NN_RATE = 0.01
        self.GOAL_THRESH = 0.5
        self.control_timer = rospy.Timer(rospy.Duration(self.CONTROL_RATE), self.cb_control)
        self.nn_timer = rospy.Timer(rospy.Duration(self.NN_RATE), self.cb_compute_action)

    def cb_compute_action(self, event):
        # Set robot
        start_time = time.time()
        px = self.pos.x
        py = self.pos.y
        vx = self.vel.x
        vy = self.vel.y
        gx = self.goal.pose.position.x
        gy = self.goal.pose.position.y
        theta = self.psi
        radius = self.robot_data['radius']
        v_pref = self.robot_pref_speed
        self.robot.set(px, py, gx, gy, vx, vy, theta, radius, v_pref)

        robot_dist_to_goal = np.linalg.norm(np.array([px, py]) - np.array([gx, gy]))

        if robot_dist_to_goal < self.GOAL_THRESH:
            self.stop_moving_flag = True
        else:
            self.stop_moving_flag = False

        # Communicate actions to the control node:A
        print "\n================================================================\n"
        print "SARLPolicyNode.compute_action:"
        print "--->self.other_agents_state obj: ", self.other_agents_state
        print "--->self.other_agents_state pos: ", self.other_agents_state[0].position
        print "--->self.other_agents_state vel: ", self.other_agents_state[0].velocity
        self.desired_action = self.robot.act(self.other_agents_state)
        print "\n--->SARLPolicyNode.compute_action runtime: ", time.time() - start_time
        print "\n================================================================\n"

    def cb_control(self, event):
        if self.stop_moving_flag:
            self.stop_moving()
            return
        else:
            twist = Twist()
            twist.linear.x = self.desired_action.vx
            twist.linear.y = self.desired_action.vy
            self.pub_twist.publish(twist)
            return

    def moving_flag_publisher(self):
        msg = Bool()
        msg.data = self.stop_moving_flag
        self.flag_pub.publish(msg)

    def cb_pose(self, msg):
        start_time = time.time()
        q = msg.pose.orientation
        _, _, yaw = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.psi = wrap(yaw)
        self.pos = msg.pose.position
        self.vel = msg.twist.linear
        #print "==================="
        #print " pose callback runtime: ", time.time() - start_time
        #print "==================="
        # TODO: pass arguments whether running in real (execute code below)
        #self.vel.x = (msg.pose.position.x - self.pose.pose.position.x)/(1.0/120)
        #self.vel.y = (msg.pose.position.y - self.pose.pose.position.y)/(1.0/120)
        #self.pose = msg

    def cb_dynamic_goal(self, msg):
        new_goal = PoseStamped()
        new_goal.pose.position.x = msg.pose.position.x
        new_goal.pose.position.y = msg.pose.position.y
        self.goal = new_goal

    def cb_other_agents(self, msg):
        # Create list of HUMANS
        # NOT: ObservableState() that represents the observable state of each obstacle/other agent
        other_agents_observable_states = []
        num_agents = 1
        for i in range(num_agents):
            radius = 0.3  # Spheres in gazebo
            x = msg.pose.position.x
            y = msg.pose.position.y
            vx = 0.01 #msg.twist[i].linear.x
            vy = 0.01 #msg.twist[i].linear.y
            other_agents_observable_states.append(ObservableState(x, y, vx, vy, radius))

        self.other_agents_state = other_agents_observable_states

    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)

    def on_shutdown(self):
        rospy.loginfo("[%s] Shutting down." % self.node_name)
        self.stop_moving()
        rospy.loginfo("Stopped %s's velocity." % self.robot_name)


def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def run(framework):
    if framework == 'SARL':
        print('Running SARL from sarl_to_twist.py')
        robot_data = {'goal': None, 'radius': 0.3, 'pref_speed': 0.3, 'kw': 10.0, 'kp': 1.0, 'name': 'balabot'}
        sarl_policy_node = SARLPolicyNode(robot_data)
        while(True):
            rospy.Duration(secs=200)

        rospy.on_shutdown(sarl_policy_node.on_shutdown)
        rospy.spin()


if __name__ == '__main__':
    print('About to run SARL')
    run('SARL')
