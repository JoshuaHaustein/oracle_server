import pickle

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from oracle_pb2 import (
    ActionResponse,
    FeasibilityResponse,
    PushabilityResponse
)


def load_models(path):
    with open(path, 'rb') as f:
        model_dict = pickle.load(f)

    x_size = model_dict['n_features_in']
    y_size = model_dict['n_features_out']
    n_hidden_units = model_dict['n_hidden_units']
    n_residual_units = model_dict['n_residual_units']

    norm = Normalization(x_size).cuda()
    norm.load_state_dict(model_dict['norm'])
    norm_inv = NormalizationInverse(y_size).cuda()
    norm_inv.load_state_dict(model_dict['norm_inv'])

    mean = torch.nn.Sequential(
        torch.nn.Linear(x_size, n_hidden_units),
        *[Residual(n_hidden_units) for _ in range(n_residual_units)],
        torch.nn.Linear(n_hidden_units, y_size)
    ).cuda()
    mean.eval()
    mean.load_state_dict(model_dict['mean_model'])

    std = torch.nn.Sequential(
        torch.nn.Linear(x_size, n_hidden_units),
        *[Residual(n_hidden_units) for _ in range(n_residual_units)],
        torch.nn.Linear(n_hidden_units, y_size),
        torch.nn.Softplus()
    ).cuda()
    std.eval()
    std.load_state_dict(model_dict['std_model'])

    return mean, std, norm, norm_inv


def features(ar):
    res = []
    if hasattr(ar, 'robot_x'):
        res.extend([
            ar.robot_x - ar.object_x,
            ar.robot_y - ar.object_y,
            np.cos(ar.robot_radians),
            np.sin(ar.robot_radians),
            ])

    # Object initial and successor state always present
    object_angle_change = np.arctan2(
        np.sin(ar.object_radians_prime - ar.object_radians),
        np.cos(ar.object_radians_prime - ar.object_radians)
    )
    res.extend([
        np.cos(ar.object_radians),
        np.sin(ar.object_radians),
        ar.object_x_prime - ar.object_x,
        ar.object_y_prime - ar.object_y,
        object_angle_change,
        ar.object_mass,
        ar.object_rotational_inertia,
        ar.object_friction,
        ar.object_width,
        ar.object_height
    ])
    res_np = np.array([res])
    return Variable(torch.cuda.FloatTensor(res_np))



class Normalization(torch.nn.Module):
    def __init__(self, n_features):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.zeros(n_features))
        self.register_buffer('std', torch.ones(n_features))

    def forward(self, x):
        return (x - Variable(self.mean)) / Variable(self.std)


class NormalizationInverse(torch.nn.Module):
    def __init__(self, n_features):
        super(NormalizationInverse, self).__init__()
        self.register_buffer('mean', torch.zeros(n_features))
        self.register_buffer('std', torch.zeros(n_features))

    def forward(self, x):
        return x * Variable(self.std) + Variable(self.mean)


class Residual(torch.nn.Module):
    
    def __init__(self, num_features):
        super(Residual, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(num_features)
        self.fc1 = torch.nn.Linear(num_features, num_features)
        self.bn2 = torch.nn.BatchNorm1d(num_features)
        self.fc2 = torch.nn.Linear(num_features, num_features)
        
    def forward(self, x):
        a = self.fc1(F.relu(self.bn1(x)))
        b = self.fc2(F.relu(self.bn2(a)))
        return b + x


class Pushability:

    def __init__(self):
        (
            self.mean,
            self.std,
            self.norm,
            self.norm_inv
        ) = load_models('saved_models/pushability_models.pkl')

        # For mahalanobis, the queried robot state needs normalizing
        self.y_norm = Normalization(3).cuda()
        self.y_norm.load_state_dict(self.norm_inv.state_dict())

    def mahalanobis(self, request):
        XY = features(request)
        # XY is:
        # [0, 1] - cosine and sine of initial object rotation
        # [2, 3] - translation of object x, y
        # [4] - change of rotation
        # [5, 6, 7, 8, 9] - object properties
        X_ = XY[:, 5:]
        Y_ = XY[:, 2:5]
        X = X_ * 0.0
        Y = self.y_norm(Y_)
        µ = self.mean(X)
        σ = self.std(X)
        response = PushabilityResponse()
        response.mahalanobis = torch.norm((µ - Y) / σ).cpu().data[0]

        # These are not relevant here
        response.projected_object_x = 0.0
        response.projected_object_y = 0.0
        response.projected_object_radians = 0.0
        return response

    def projection(self, request):
        XY = features(request)
        n_stds = request.num_stds
        # XY is:
        # [0, 1] - cosine and sine of initial object rotation
        # [2, 3] - translation of object x, y
        # [4] - change of rotation
        # [5, 6, 7, 8, 9] - object properties
        X_ = XY[:, 5:]
        Y_ = XY[:, 2:5]
        X = X_ * 0.0
        Y = self.y_norm(Y_)
        µ = self.mean(X)
        σ = self.std(X)
        mahalanobis = torch.norm((µ - Y) / σ)
        y = self.norm_inv(µ + (Y - µ) / mahalanobis * n_stds).cpu().data

        response = PushabilityResponse()
        # This is not relevant now
        response.mahalanobis = -1

        # Actual response
        response.projected_object_x = request.object_x + y[0, 0]
        response.projected_object_y = request.object_y + y[0, 1]
        response.projected_object_radians = request.object_radians + y[0, 2]
        return response


class Feasibility:

    def __init__(self):
        (
            self.mean,
            self.std,
            self.norm,
            self.norm_inv
        ) = load_models('saved_models/feasibility_models.pkl')

        # For mahalanobis, the queried robot state needs normalizing
        self.y_norm = Normalization(4).cuda()
        self.y_norm.load_state_dict(self.norm_inv.state_dict())

    def sample(self, request):
        XY = features(request)
        X = self.norm(XY[:, 4:])
        µ = self.mean(X)
        σ = self.std(X)
        z = Variable(torch.randn(1, 4)).cuda()
        y = self.norm_inv(µ + z * σ).cpu().data
        response = FeasibilityResponse()
        response.mahalanobis = -1.0 # Not valid here
        response.robot_x = request.object_x + y[0, 0]
        response.robot_y = request.object_y + y[0, 1]
        response.robot_radians = np.arctan2(y[0, 3], y[0, 2])
        return response

    def mahalanobis(self, request):
        XY = features(request)
        X = self.norm(XY[:, 4:])
        µ = self.mean(X).cpu().data.numpy().flatten()
        σ = self.std(X).cpu().data.numpy().flatten()
        y = self.y_norm(XY[:, :4]).cpu().data.numpy().flatten()
        response = FeasibilityResponse()
        response.mahalanobis = np.linalg.norm((y - µ) / σ)
        response.robot_x = 0.0          # Ignored
        response.robot_y = 0.0          #  -"-
        response.robot_radians = 0.0    #  -"-
        return response


class Oracle:

    def __init__(self):
        (
            self.mean,
            self.std,
            self.norm,
            self.norm_inv
        ) = load_models('saved_models/oracle_models.pkl')

    def sample(self, action_request):
        ar = action_request
        X = self.norm(features(ar))
        µ = self.mean(X)
        σ = self.std(X)
        z = Variable(torch.randn(µ.size())).cuda()
        y = self.norm_inv(z * σ + µ).cpu().data
        action_response = ActionResponse()
        action_response.dx = y[0, 0]
        action_response.dy = y[0, 1]
        action_response.dr = y[0, 2]
        action_response.t = y[0, 3]
        return action_response


if __name__ == '__main__':
    from oracle_pb2 import PushabilityRequest
    oracle = Pushability()
    request = PushabilityRequest()
    #request.robot_x = 9.8;
    #request.robot_y = 9.8;
    #request.robot_radians = -0.92;
    for n_stds in [0, 1, 2, 3]:
        print('n_stds:', n_stds)
        request.num_stds = n_stds
        request.object_x = 10.0
        request.object_y = 10.0
        request.object_radians = np.pi / 4.0

        request.object_x_prime = 15.0
        request.object_y_prime = 15.0
        request.object_radians_prime = np.pi / 4.0

        request.object_mass = 0.51
        request.object_rotational_inertia = 0.008
        request.object_friction = 0.126
        request.object_width = 0.144
        request.object_height = 0.144
        print(oracle.projection(request))
