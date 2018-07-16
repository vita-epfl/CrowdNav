from gym_crowd.envs.policy.orca import ORCA
from gym_crowd.envs.utils.state import FullState, ObservableState, JointState
from gym_crowd.envs.utils.action import ActionXY
import numpy as np


def test_orca():
    self_state = FullState(0, -2, 0, 1, 0.3, 0, 2, 1, 0)
    ped_state = ObservableState(-1, 1, 1, 0, 0.3)
    state = JointState(self_state, [ped_state])

    orca = ORCA()
    action = orca.predict(state)
    assert np.allclose(action, ActionXY(0, 1))
