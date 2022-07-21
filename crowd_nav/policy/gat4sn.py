'''
Graph Attention Network for Social Navigation (GAT4SN)
'''
from torch import nn
import torch
import logging
import torch.nn.functional as F
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from crowd_nav.policy.cadrl import mlp

class GraphAttentionLayerForSingleNode(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, concat = True):
        super(GraphAttentionLayerForSingleNode, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size = (in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain = 1.414)
        self.a = nn.Parameter(torch.empty(size = (2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)
        self.attention_weights = None
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        '''
        Dimension of h = B x N x D
        B = batch_size
        N = 1 (robot itself) + num_humans + num_obstacles
        D = in_dim
        O = out_dim
        '''
        B, N, D = h.shape
        # print("B = {}".format(B))
        # print("N = {}".format(N))
        # print("D = {}".format(D))
        Wh = torch.mm(h.view(B * N, -1), self.W) ## Wh.shape = (B, N, O)
        Wh = Wh.view(B, N, -1)
        assert Wh.shape == (B, N, self.out_dim)
        robot_state = Wh[:, 0, :]
        assert robot_state.shape == (B, self.out_dim)
        other_agent_states = Wh[:, 1:, :]
        assert other_agent_states.shape == (B, N - 1, self.out_dim)
        cat1 = torch.cat((robot_state.view(B, 1, -1).expand(B, N - 1, -1), other_agent_states), dim = -1)
        assert cat1.shape == (B, N - 1, 2 * self.out_dim)
        attention = torch.mm(cat1.view(B * (N - 1), -1), self.a).squeeze()
        attention = attention.view(B, N - 1)
        assert attention.shape == (B, N - 1)
        attention = F.softmax(attention, dim = 1)
        self.attention_weights = attention.squeeze().data.cpu().numpy()
        # print("attention's shape = {}".format(attention.shape))
        # print("other_agent_states's shape = {}".format(other_agent_states.shape))
        h_prime = attention.unsqueeze(-1) * other_agent_states
        # h_prime = torch.matmul(attention, other_agent_states)
        # print("h_prime's shape = {}".format(h_prime.shape))
        assert h_prime.shape == (B, N - 1, self.out_dim)

        h_prime_cat = torch.cat((robot_state.view(B, 1, -1), h_prime), dim = 1)
        if self.concat:
            return F.elu(h_prime_cat)
        else:
            return h_prime_cat

class GAT(nn.Module):
    def __init__(self, num_feat, num_hidden_feat, num_out_feat, alpha, nheads):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayerForSingleNode(num_feat, num_hidden_feat, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerForSingleNode(num_hidden_feat * nheads, num_out_feat, alpha=alpha, concat=False)

    def forward(self, x):
        x = torch.cat([att(x) for att in self.attentions], dim = -1)
        x = F.elu(self.out_att(x))
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, num_hidden_feat, num_out_feat, num_heads, mlp1_dims, mlp2_dims, mlp3_dims, alpha):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.other_agent_states_dim = input_dim - self_state_dim
        if mlp1_dims[-1] != mlp2_dims[-1]:
           raise ValueError('The output for mlp1 and mlp should have same dimension') 
        self.mlp1 = mlp(self_state_dim, mlp1_dims, last_relu = True)
        self.mlp2 = mlp(self.other_agent_states_dim, mlp2_dims, last_relu = True)
        self.mlp3 = mlp(2 * num_out_feat, mlp3_dims, last_relu = False)
        self.gat = GAT(mlp1_dims[-1], num_hidden_feat, num_out_feat, alpha, num_heads)


    def forward(self, state):
        '''
        Dimension of state = B x N x D
        B = batch_size
        N = num_humans + num_obstacles
        D = robot_state_dim + other_agent_states_dim
        '''
        B, N, D = state.shape

        robot_state = state[:, 0, :self.self_state_dim]
        assert robot_state.shape == (B, self.self_state_dim)
        robot_state_embed = self.mlp1(robot_state)

        other_agent_states = state[:, :, self.self_state_dim:]
        assert other_agent_states.shape == (B, N, self.other_agent_states_dim)
        other_agent_states_embed = self.mlp2(other_agent_states.view(-1, self.other_agent_states_dim))
        other_agent_states_embed = other_agent_states_embed.view(B, N, -1)

        embed_cat = torch.cat((robot_state_embed.view(B, 1, -1), other_agent_states_embed), dim = 1)

        gat_embed = self.gat(embed_cat)

        robot_gat_embed = gat_embed[:, 0, :]
        other_agent_gat_embeds = gat_embed[:, 1:, :]
        other_agent_gat_embeds = other_agent_gat_embeds.sum(dim = 1)

        cat_embed = torch.cat((robot_gat_embed, other_agent_gat_embeds), dim = -1)
        value = self.mlp3(cat_embed)

        return value



class GAT4SN(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'GAT4SN'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('gat4sn', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('gat4sn', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('gat4sn', 'mlp3_dims').split(', ')]
        num_hidden_feature = mlp1_dims[-1]
        num_out_feat = int(mlp3_dims[0] / 2)
        num_heads = config.getint('gat4sn', 'num_heads')
        alpha = config.getfloat('gat4sn', 'alpha')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, num_hidden_feature, num_out_feat, num_heads, mlp1_dims, mlp2_dims, mlp3_dims, alpha)
        self.multiagent_training = config.getboolean('gat4sn', 'multiagent_training')
        self.num_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logging.info('Policy: {} with {} attention heads'.format(self.name, num_heads))
        logging.info('Number of parameters: {}'.format(self.num_total_params))

    def get_attention_weights(self):
        return self.model.gat.out_att.attention_weights