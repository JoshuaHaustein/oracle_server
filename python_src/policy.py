import pickle

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class Normalization(torch.nn.Module):
    def __init__(self, n_features):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.zeros(n_features))
        self.register_buffer('std', torch.ones(n_features))

    def forward(self, x):
        return (x - Variable(self.mean)) / Variable(self.std)



class Policy(torch.nn.Module):
    def __init__(self, x_size, g_size, hidden_size=64):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(x_size + g_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_xy_vel = torch.nn.Linear(hidden_size, 1)
        self.fc_xy_dir = torch.nn.Linear(hidden_size, 2)
        self.fc_r_vel = torch.nn.Linear(hidden_size, 1)
        self.fc_dur = torch.nn.Linear(hidden_size, 1)
        self.register_buffer(
            'xy_vel_scale',
            torch.FloatTensor([0.6])
        )
        self.register_buffer(
            'r_vel_scale',
            torch.FloatTensor([1.4])
        )
        self.register_buffer(
            'dur_scale',
            torch.FloatTensor([0.6])
        )

    def forward(self, x, g):
        xg = torch.cat([x, g], dim=1)
        a1 = F.relu(self.fc1(xg))
        a2 = F.relu(self.fc2(a1))
        u_xy_vel = F.sigmoid(self.fc_xy_vel(a2))
        u_xy_dir = F.tanh(self.fc_xy_dir(a2))
        u_r_vel = F.tanh(self.fc_r_vel(a2))
        u_dur = F.sigmoid(self.fc_dur(a2))
        return torch.cat(
            [
                Variable(self.xy_vel_scale) * u_xy_vel * u_xy_dir / u_xy_dir.norm(dim=1, keepdim=True),
                Variable(self.r_vel_scale) * u_r_vel,
                Variable(self.dur_scale) * u_dur,
                ],
            dim=1
        )


class ProductionPolicy(torch.nn.Module):
    def __init__(self, x_size, g_size):
        super(ProductionPolicy, self).__init__()
        self.norm_x = Normalization(x_size)
        self.norm_y = Normalization(g_size)
        self.policy = Policy(x_size, g_size)

    def forward(self, x, g):
        return self.policy(self.norm_x(x), self.norm_y(g))
