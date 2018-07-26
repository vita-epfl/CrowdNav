from gym_crowd.envs.policy.linear import LinearPolicy
from gym_crowd.envs.policy.orca import ORCA
from gym_crowd.envs.policy.trajnet import Trajnet


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = LinearPolicy
policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['trajnet'] = Trajnet
