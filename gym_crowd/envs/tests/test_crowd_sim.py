import configparser
import gym
from gym_crowd.envs.utils.action import ActionXY


def test_crowd_sim():
    config = configparser.ConfigParser()
    config['env'] = {'num_peds': 2}
    config['peds'] = {'visible': False, 'policy': 'linear', 'radius': 0.3, 'v_pref': 1,
                      'sensor': 'Coordinates', 'kinematics': False}
    config['navigator'] = {'visible': True, 'policy': 'linear', 'radius': 0.3, 'v_pref': 1,
                           'sensor': 'Coordinates', 'kinematics': False}

    env = gym.make('CrowdSim-v0')
    env.configure(config)

    # failure case
    env.reset()
    action = ActionXY(0, 1)
    ob, reward, done = env.step(action)
    assert reward == -0.25

    # success case
    env.reset()
    env.step(ActionXY(0, 0.1))
    env.step(ActionXY(0, 0.1))
    ob, reward, done = env.step(ActionXY(0, 3.8))
    assert reward == 1


test_crowd_sim()