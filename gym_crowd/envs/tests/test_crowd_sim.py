import configparser
import gym
from gym_crowd.envs.utils.action import ActionXY
from dynav.utils.navigator import Navigator


def test_crowd_sim():
    config = configparser.ConfigParser()
    config['env'] = {'num_peds': 2}
    config['peds'] = {'visible': False, 'policy': 'linear', 'radius': 0.3, 'v_pref': 1,
                      'sensor': 'Coordinates', 'kinematics': False}
    config['navigator'] = {'visible': True, 'policy': 'linear', 'radius': 0.3, 'v_pref': 1,
                           'sensor': 'Coordinates', 'kinematics': False}

    env = gym.make('CrowdSim-v0')
    env.configure(config)
    navigator = Navigator(config, 'navigator')
    navigator.policy = None
    env.set_navigator(navigator)

    # failure case
    env.reset()
    action = ActionXY(0, 1)
    ob, reward, done, info = env.step(action)
    assert reward == -0.25

    # success case
    env.reset()
    env.step(ActionXY(0, 0.1))
    env.step(ActionXY(0, 0.1))
    ob, reward, done, info = env.step(ActionXY(0, 3.8))
    assert reward == 1
