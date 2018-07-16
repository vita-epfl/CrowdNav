from collections import namedtuple

FullState = namedtuple('FullState', ['px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'])
ObservableState = namedtuple('ObservableState', ['px', 'py', 'vx', 'vy', 'radius'])


class JointState(object):
    def __init__(self, self_state, ped_states):
        assert isinstance(self_state, FullState)
        for ped_state in ped_states:
            assert isinstance(ped_state, ObservableState)

        self.self_state = self_state
        self.ped_states = ped_states
