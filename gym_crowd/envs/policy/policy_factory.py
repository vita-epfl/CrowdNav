from gym_crowd.envs.policy.linear import LinearPolicy
from dynav.policy.value_network import ValueNetworkPolicy

policy_factory = dict({'linear': LinearPolicy, 'value_network': ValueNetworkPolicy})
