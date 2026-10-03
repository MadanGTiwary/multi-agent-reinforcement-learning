from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module, ABC):

    def __init__(self, input_shape: int, n_actions: int, rnn_hidden_dim: int):
        super(RNN, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, n_actions)

    def forward(self, obs, hidden_state):
        x = torch.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMixNet(nn.Module, ABC):
    def __init__(self, n_agents: int, state_shape: int):
        super(QMixNet, self).__init__()
        self.qmix_hidden_dim = 32
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.hyper_w1 = nn.Linear(state_shape, n_agents * self.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(state_shape, self.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(state_shape, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1)
                                      )

    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
