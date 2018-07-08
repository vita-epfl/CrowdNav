from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='gym_crowd.envs:CrowdSim',
)