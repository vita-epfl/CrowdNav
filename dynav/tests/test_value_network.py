import configparser
from dynav.policy.value_network import ValueNetwork, ValueNetworkPolicy
import torch
import numpy as np


def test_value_network():
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


# def test_value_network_policy():
#     config = configparser.ConfigParser()
#     config['value_network'] = {'reparameterization': True, 'state_dim': 15, 'gamma': 0.9,
#                                'kinematics': 'holonomic', 'discrete': True, 'fc_layers': '150, 100, 100'}
#     policy = ValueNetworkPolicy()
#     policy.configure(config)
#