from gym_crowd.envs.policy.linear import LinearPolicy
from gym_crowd.envs.policy.orca import ORCA

policy_factory = dict({'linear': LinearPolicy, 'orca': ORCA})
