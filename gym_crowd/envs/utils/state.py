from collections import namedtuple

# FullState = namedtuple('FullState', ['px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'])
# ObservableState = namedtuple('ObservableState', ['px', 'py', 'vx', 'vy', 'radius'])


class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)


class JointState(object):
    def __init__(self, self_state, ped_states):
        assert isinstance(self_state, FullState)
        for ped_state in ped_states:
            assert isinstance(ped_state, ObservableState)

        self.self_state = self_state
        self.ped_states = ped_states
