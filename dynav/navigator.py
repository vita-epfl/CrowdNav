from gym_crowd.envs.utils.agent import Agent
from gym_crowd.envs.utils.state import State


class Navigator(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        state = State(self.px, self.py, self.gx, self.gy, self.v_pref, self.radius, ob)
        action = self.policy.predict(state, self.kinematics)
        return action
