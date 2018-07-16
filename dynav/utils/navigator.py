from gym_crowd.envs.utils.agent import Agent
from gym_crowd.envs.utils.state import JointState


class Navigator(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
