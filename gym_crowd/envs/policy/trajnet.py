import numpy as np
from gym_crowd.envs.policy.policy import Policy
from gym_crowd.envs.utils.action import ActionXY


class Trajnet(Policy):
    def __init__(self):
        """
        Every scene is a list of paths and every path is a list of `TrackRow` which has `frame`, `pedestrian`,
        `x` and `y` attributes. Frame rate is about 2.5 rows per second.

        """
        super().__init__()
        self.trajectory = None

    def configure(self, config):
        self.trajectory = config

    def predict(self, state):
        px = state.self_state.px
        py = state.self_state.py
        action = None
        for time in range(len(self.trajectory)):
            if np.allclose((self.trajectory[time].x,self.trajectory[time].y), (px, py)):
                if time == len(self.trajectory) - 1:
                    action = ActionXY(0, 0)
                else:
                    vx = self.trajectory[time+1].x - self.trajectory[time].x
                    vy = self.trajectory[time+1].y - self.trajectory[time].y
                    action = ActionXY(vx, vy)
                break
        assert action is not None

        return action
