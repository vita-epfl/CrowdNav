import numpy as np
import socialforce
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class SocialForce(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'SocialForce'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """

        :param state:
        :return:
        """
        sf_state = []
        self_state = state.self_state

        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        sf_state.append((self_state.px, self_state.py, pref_vel[0], pref_vel[1], self_state.gx, self_state.gy))
        for human_state in state.human_states:
            sf_state.append((human_state.px, human_state.py, human_state.vx, human_state.vy, 0, 0))
        sim = socialforce.Simulator(np.array(sf_state), delta_t=self.time_step)
        sim.step()
        action = ActionXY(sim.state[0, 2], sim.state[0, 3])

        self.last_state = state

        return action


class CentralizedSocialForce(SocialForce):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        sf_state = []
        for agent_state in state:
            velocity = np.array((agent_state.gx - agent_state.px, agent_state.gy - agent_state.py))
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > 1 else velocity
            sf_state.append((agent_state.px, agent_state.py, pref_vel[0], pref_vel[1], agent_state.gx, agent_state.py))

        sim = socialforce.Simulator(np.array(sf_state), delta_t=self.time_step)
        sim.step()
        actions = [ActionXY(sim.state[i, 2], sim.state[i, 3]) for i in range(len(state))]

        return actions
