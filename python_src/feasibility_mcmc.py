import pickle

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class RewardFCPlain(torch.nn.Module):
    def __init__(self, x_size, u_size, y_size, hidden_units=64):
        super(RewardFCPlain, self).__init__()

        self.fc1 = torch.nn.Linear(x_size + u_size, hidden_units)
        self.bn1 = torch.nn.BatchNorm1d(hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, hidden_units)
        self.bn2 = torch.nn.BatchNorm1d(hidden_units)
        self.fcy = torch.nn.Linear(hidden_units, y_size)  # E[Y]

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        a1 = F.relu(self.bn1(self.fc1(xu)))
        a2 = F.relu(self.bn2(self.fc2(a1)))
        y = self.fcy(a2)  # E(Y)
        # Var(Y) = E(Y^2) - E(Y)^2
        # E(Y^2) = Var(Y) + E(Y)^2
        # y2 = F.softplus(self.fcy2(a2)) + y ** 2 # E(Y^2)
        # t = F.softplus(self.fct(a2))
        return y


class FCPositive(torch.nn.Module):
    def __init__(self, x_size, u_size, y_size, hidden_units=64):
        super(FCPositive, self).__init__()

        self.fc1 = torch.nn.Linear(x_size + u_size, hidden_units)
        self.bn1 = torch.nn.BatchNorm1d(hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, hidden_units)
        self.bn2 = torch.nn.BatchNorm1d(hidden_units)
        self.fcy = torch.nn.Linear(hidden_units, y_size)  # E[Y]

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        a1 = F.relu(self.bn1(self.fc1(xu)))
        a2 = F.relu(self.bn2(self.fc2(a1)))
        y = F.softplus(self.fcy(a2))
        return y


class Normalization(torch.nn.Module):
    def __init__(self, n_features):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.zeros(n_features))
        self.register_buffer('std', torch.ones(n_features))

    def forward(self, x):
        return (x - Variable(self.mean)) / Variable(self.std)


class ExpectedDistanceProduction(torch.nn.Module):
    def __init__(self, policy, mean_model=None, variance_model=None, x_norm=None, u_norm=None, y_norm=None):
        super(ExpectedDistanceProduction, self).__init__()
        from policy import Policy
        self.policy = Policy(5, 3)
        self.mean = RewardFCPlain(5, 4, 3)
        self.variance = FCPositive(5, 4, 3)
        self.x_norm = Normalization(5)
        self.u_norm = Normalization(4)
        self.g_norm = Normalization(3)
        self.register_buffer(
            'weights',
            torch.FloatTensor([1.0, 1.0, 0.1])
        )

    def forward(self, x, g):
        """
        Goal is relative object! G_rel = (G - X)
        """
        x = self.x_norm(x)
        g = self.g_norm(g)
        u = self.u_norm(self.policy(x, g))
        mean = self.mean(x, u)
        var = self.variance(x, u)
        return ((var + (mean - g) ** 2) * Variable(self.weights)).mean(dim=1)


def _proposal2state(θ_obj, θ_rob, x_rob, y_rob, dx, dy, dθ):
    obj_x = np.cos(θ_rob) * (-x_rob) + np.sin(θ_rob) * (-y_rob)
    obj_y = -np.sin(θ_rob) * (-x_rob) + np.cos(θ_rob) * (-y_rob)
    obj_θ = θ_obj - θ_rob
    goal_x = np.cos(θ_rob) * dx + np.sin(θ_rob) * dy
    goal_y = -np.sin(θ_rob) * dx + np.cos(θ_rob) * dy

    return obj_x, obj_y, obj_θ, goal_x, goal_y, dθ


def _energy(θ_obj, θ_rob_prop, x_rob_prop, y_rob_prop, dx, dy, dθ, width, height, expect_distance):
    obj_x, obj_y, obj_θ, goal_x, goal_y, dθ = _proposal2state(θ_obj, θ_rob_prop, x_rob_prop, y_rob_prop, dx, dy, dθ)
    X = Variable(torch.FloatTensor([[obj_x, obj_y, obj_θ, width, height]]))
    G = Variable(torch.FloatTensor([[goal_x, goal_y, dθ]]))
    U = expect_distance(X, G)
    return U.cpu().data[0]


# Just give change of state, (and object width, height)
# Then sample different poses of robot (as markov chain) and rotate/translate object accordingly
# Run k times and return last sample
def mcmc(θ_obj, dx, dy, dθ, width, height, expect_distance):
    # select initial robot position relative object
    θ_rob = np.arctan2(dy, dx) - np.pi / 2.0
    x_rob = -0.25 * np.cos(θ_rob + np.pi / 2.0)
    y_rob = -0.25 * np.sin(θ_rob + np.pi / 2.0)

    U = _energy(θ_obj, θ_rob, x_rob, y_rob, dx, dy, dθ, width, height, expect_distance)

    while True:
        θ_rob_prop = θ_rob + np.random.randn() * 0.5
        θ_rob_prop = np.arctan2(np.sin(θ_rob_prop), np.cos(θ_rob_prop))
        x_rob_prop = x_rob + np.random.randn() * 0.1
        y_rob_prop = y_rob + np.random.randn() * 0.1

        U_ = _energy(θ_obj, θ_rob_prop, x_rob_prop, y_rob_prop, dx, dy, dθ, width, height, expect_distance)
        T = 1e-2
        p = min(1, np.exp((U - U_) / T))
        if np.random.rand() < p:
            θ_rob = θ_rob_prop
            x_rob = x_rob_prop
            y_rob = y_rob_prop
            yield x_rob, y_rob, θ_rob

