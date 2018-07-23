from gym_crowd.envs.policy.linear import LinearPolicy
from gym_crowd.envs.policy.orca import ORCA
from gym_crowd.envs.policy.trajnet import Trajnet


def none_policy():
    return None


policy_factory = dict({'linear': LinearPolicy, 'orca': ORCA, 'none': none_policy, 'trajnet': Trajnet})
