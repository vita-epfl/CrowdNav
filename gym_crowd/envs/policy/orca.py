import numpy as np
from gym_crowd.envs.policy.policy import Policy
from gym_crowd.envs.utils.action import ActionXY
import rvo2


class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer he running
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

        """
        super().__init__()
        self.trainable = False
        self.time_step = 1
        self.neighbor_dist = 1.5
        self.max_neighbors = 5
        self.time_horizon = 1.5
        self.time_horizon_obst = 2
        self.radius = 0.4
        self.max_speed = 2

    def configure(self, config):
        # self.time_step = config.getfloat('orca', 'time_step')
        # self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        # self.max_neighbors = config.getint('orca', 'max_neighbors')
        # self.time_horizon = config.getfloat('orca', 'time_horizon')
        # self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        # self.radius = config.getfloat('orca', 'radius')
        # self.max_speed = config.getfloat('orca', 'max_speed')
        return

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step

        :param state:
        :return:
        """
        if self.reach_destination(state):
            return ActionXY(0, 0)

        self_state = state.self_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
        self_agent = sim.addAgent(self_state.position, *params, self_state.radius, self_state.v_pref, self_state.velocity)
        other_agents = [sim.addAgent(ped_state.position, *params, ped_state.radius, self.max_speed, ped_state.velocity)
                        for ped_state in state.ped_states]

        # set preferred velocity
        theta = np.arctan2(self_state.gy - self_state.py, self_state.gx - self_state.px)
        pref_vel = (np.cos(theta) * self_state.v_pref, np.sin(theta) * self_state.v_pref)
        sim.setAgentPrefVelocity(self_agent, pref_vel)

        # TODO: make sure setting preferred velocity for other peds doesn't affect the computing of self agent
        for i, ped_state in enumerate(state.ped_states):
            pref_vel = (1, 1)
            sim.setAgentPrefVelocity(other_agents[i], pref_vel)

        sim.doStep()
        next_position = sim.getAgentPosition(self_agent)
        action = ActionXY(next_position[0]-self_state.px, next_position[1]-self_state.py)

        return action
