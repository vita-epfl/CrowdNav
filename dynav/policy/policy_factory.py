from gym_crowd.envs.policy.policy_factory import policy_factory
from dynav.policy.cadrl import CADRL

policy_factory['cadrl'] = CADRL
