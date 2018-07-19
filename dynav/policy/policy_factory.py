from gym_crowd.envs.policy.policy_factory import policy_factory
from dynav.policy.value_network import ValueNetworkPolicy

policy_factory['value_network'] = ValueNetworkPolicy
