from gym_crowd.envs.utils.agent import Agent
from gym_crowd.envs.utils.state import State


class Pedestrian(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        """
        The state for pedestrian is its and all other pedestrians' position and goals
        :param ob:
        :return:
        """
        state = State(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
