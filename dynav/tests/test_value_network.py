import configparser
from dynav.policy.cadrl import ValueNetwork, CADRL
import torch
import numpy as np


def test_cadrl():
    vn = ValueNetwork(True, 15, 'holonomic', [150, 100, 100])
    state = (2, 2, 0, 1, 0.3, 2, 4, 1, 0, 4, 2, 2, 0, 0.3)
    state = torch.Tensor(state).expand(1, 14)
    rotated_state = vn.rotate(state).squeeze().numpy()
    assert np.allclose(rotated_state, [2, 1, 1, 0, 0.3, 0, 0, -2, 0, -2, 0.3, 0.6, 1, 0, 2], atol=1e-06)

    vn = ValueNetwork(True, 14, 'unicycle', [150, 100, 100])
    state = (2, 2, 0, 1, 0.3, 2, 4, 1, 0, 4, 2, 2, 0, 0.3)
    state = torch.Tensor(state).expand(1, 14)
    rotated_state = vn.rotate(state).squeeze().numpy()
    assert np.allclose(rotated_state, [2, 1, 1, 0, 0.3, -np.pi/2, 0, -2, 0, -2, 0.3, 0.6, 0, -1, 2], atol=1e-06)
