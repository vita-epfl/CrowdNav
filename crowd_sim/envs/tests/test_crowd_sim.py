import configparser
import gym
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.utils.robot import Robot


def test_crowd_sim():
    config = configparser.ConfigParser()
    config['env'] = {'train_human_num': 2, 'time_limit': 25}
    config['humans'] = {'visible': True, 'policy': 'linear', 'radius': 0.3, 'v_pref': 1,
                      'sensor': 'coordinates', 'kinematics': 'holonomic'}
    config['robot'] = {'visible': True, 'policy': 'linear', 'radius': 0.3, 'v_pref': 1,
                           'sensor': 'coordinates', 'kinematics': 'holonomic'}

    env = gym.make('CrowdSim-v0')
    env.configure(config)
    robot = Robot(config, 'robot')
    robot.policy = None
    env.set_robot(robot)

    # failure case
    env.reset('test', test_case=0)
    action = ActionXY(0, 1)
    ob, reward, done, info = env.step(action)
    assert reward == -0.25

    # success case
    env.reset('test', test_case=0)
    env.step(ActionXY(0, 0.1))
    env.step(ActionXY(0, 0.1))
    ob, reward, done, info = env.step(ActionXY(0, 3.8))
    assert reward == 1
