from collections import namedtuple

FullState = namedtuple('FullState', ['px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'])
ObservableState = namedtuple('ObservableState', ['px', 'py', 'vx', 'vy', 'radius'])
# JointState = namedtuple('JointState', ['px', 'py', 'vx', 'vy', 'radius', 'px', 'py', 'v_pref', 'theta',
#                                        'px1', 'py1', 'vx1', 'vy1', 'radius1'])


class State(object):
    def __init__(self, self_state, ped_states):
        self.self_state = self_state
        self.ped_states = ped_states
