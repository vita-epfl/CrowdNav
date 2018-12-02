import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state_input):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state_input: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        if isinstance(state_input, tuple):
            state, lengths = state_input
        else:
            state = state_input
            lengths = torch.IntTensor([state.size()[1]])

        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.reshape((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        mask = rnn_utils.pad_sequence([torch.ones(length.item()) for length in lengths], batch_first=True)
        masked_scores = scores * mask.float()
        max_scores = torch.max(masked_scores, dim=1, keepdim=True)[0]
        exps = torch.exp(masked_scores - max_scores)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(1, keepdim=True)
        weights = (masked_exps / masked_sums).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'
        self.attention_weights = None

    def configure(self, config):
        self.set_common_parameters(config)
        self.with_om = config.sarl.with_om
        self.multiagent_training = config.sarl.multiagent_training

        mlp1_dims = config.sarl.mlp1_dims
        mlp2_dims = config.sarl.mlp2_dims
        mlp3_dims = config.sarl.mlp3_dims
        attention_dims = config.sarl.attention_dims
        with_global_state = config.sarl.with_global_state
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.attention_weights
