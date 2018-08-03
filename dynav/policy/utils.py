import numpy as np
from numpy.linalg import norm
from gym_crowd.envs.utils.utils import point_to_segment_dist


def reward(state, action, kinematics, time_step):
    """ Reward function for cadrl and srl
    Compute rewards given agent action and assuming other agents travelling at the observed speed

    :param state:
    :param action:
    :param kinematics:
    :param time_step:
    :return:
    """
    if kinematics == 'unicycle':
        raise NotImplemented
    # compute minimum distance to other agents
    self_state = state.self_state
    dmin = float('inf')
    collision = False
    for i, ped_state in enumerate(state.ped_states):
        px = ped_state.px - self_state.px
        py = ped_state.py - self_state.py
        vx = ped_state.vx - action.vx
        vy = ped_state.vy - action.vy
        ex = px + vx * time_step
        ey = py + vy * time_step
        closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0)
        if closest_dist < ped_state.radius + self_state.radius:
            collision = True
            break
        elif closest_dist < dmin:
            dmin = closest_dist

    # check if reaching the goal
    end_position = np.array(self_state.position) + np.array(action) * time_step
    reaching_goal = norm(end_position - np.array(self_state.position)) < self_state.radius

    if collision:
        reward_value = -0.25
    elif reaching_goal:
        reward_value = 1
    elif dmin < 0.2:
        reward_value = -0.1 + dmin / 2
    else:
        reward_value = 0

    return reward_value
