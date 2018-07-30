import numpy as np
import rvo2

from gym_crowd.envs.policy.policy import Policy
from gym_crowd.envs.utils.action import ActionXY


class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'orca'
        self.trainable = False
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 2
        self.time_horizon_obst = 2
        self.radius = 0.3
        self.max_speed = 1

    def configure(self, config):
        # self.time_step = config.getfloat('orca', 'time_step')
        # self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        # self.max_neighbors = config.getint('orca', 'max_neighbors')
        # self.time_horizon = config.getfloat('orca', 'time_horizon')
        # self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        # self.radius = config.getfloat('orca', 'radius')
        # self.max_speed = config.getfloat('orca', 'max_speed')
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = state.self_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
        self_agent = sim.addAgent(self_state.position, *params, self_state.radius, self_state.v_pref, self_state.velocity)
        other_agents = [sim.addAgent(ped_state.position, *params, ped_state.radius, self.max_speed, ped_state.velocity)
                        for ped_state in state.ped_states]

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        goal_direction = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        norm = np.linalg.norm(goal_direction)
        if norm > self.max_speed:
            pref_vel = goal_direction / norm if norm != 0 else np.array((0., 0.))
        else:
            pref_vel = goal_direction

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        sim.setAgentPrefVelocity(self_agent, tuple(pref_vel))
        for i, ped_state in enumerate(state.ped_states):
            pref_vel = (self.max_speed, 0)
            sim.setAgentPrefVelocity(other_agents[i], pref_vel)

        sim.doStep()
        next_position = sim.getAgentPosition(self_agent)
        action = ActionXY((next_position[0] - self_state.px) / self.time_step,
                          (next_position[1] - self_state.py) / self.time_step)

        # save state for imitation learning
        self.last_state = state

        return action
