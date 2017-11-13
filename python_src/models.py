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
from policy import ProductionPolicy


def load_models(path):
    with open(path, 'rb') as f:
        model_dict = pickle.load(f)

    x_size = model_dict['n_features_in']
    y_size = model_dict['n_features_out']
    n_hidden_units = model_dict['n_hidden_units']
    n_residual_units = model_dict['n_residual_units']

    norm = Normalization(x_size)
    norm.load_state_dict(model_dict['norm'])
    norm_inv = NormalizationInverse(y_size)
    norm_inv.load_state_dict(model_dict['norm_inv'])

    mean = torch.nn.Sequential(
        torch.nn.Linear(x_size, n_hidden_units),
        *[Residual(n_hidden_units) for _ in range(n_residual_units)],
        torch.nn.Linear(n_hidden_units, y_size)
    )
    mean.eval()
    mean.load_state_dict(model_dict['mean_model'])

    std = torch.nn.Sequential(
        torch.nn.Linear(x_size, n_hidden_units),
        *[Residual(n_hidden_units) for _ in range(n_residual_units)],
        torch.nn.Linear(n_hidden_units, y_size),
        torch.nn.Softplus()
    )
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
    return Variable(torch.FloatTensor(res_np))



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
        self.y_norm = Normalization(3)
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
        response.mahalanobis = torch.norm((µ - Y) / σ).data[0]

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
        y = self.norm_inv(µ + (Y - µ) / mahalanobis * n_stds).data

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

        self.policy = ProductionPolicy(x_size=5, g_size=3)
        with open('saved_models/production_policy.pkl', 'rb') as f:
            self.policy.load_state_dict(pickle.load(f))
        self.policy.eval()

        from feasibility_mcmc import ExpectedDistanceProduction
        self.expected_distance = ExpectedDistanceProduction(self.policy)
        with open('saved_models/expected_distance.pkl', 'rb') as f:
            self.expected_distance.load_state_dict(pickle.load(f))
        self.expected_distance.eval()

        # For mahalanobis, the queried robot state needs normalizing
        self.y_norm = Normalization(4)
        self.y_norm.load_state_dict(self.norm_inv.state_dict())

    def sample(self, request):
        dx = request.object_x_prime - request.object_x
        dy = request.object_y_prime - request.object_y
        dθ = 0.0
        from feasibility_mcmc import mcmc
        for i, (rx, ry, rθ) in enumerate(mcmc(request.object_radians, dx, dy, dθ, request.object_width, request.object_height, self.expected_distance)):
            if i > 10:
                break

        response = FeasibilityResponse()
        response.mahalanobis = -1.0 # Not valid here
        response.robot_x = request.object_x + rx
        response.robot_y = request.object_y + ry
        response.robot_radians = rθ
        return response

    def mahalanobis(self, request):
        XY = features(request)
        X = self.norm(XY[:, 4:])
        µ = self.mean(X).data.numpy().flatten()
        σ = self.std(X).data.numpy().flatten()
        y = self.y_norm(XY[:, :4]).data.numpy().flatten()
        response = FeasibilityResponse()
        response.mahalanobis = np.linalg.norm((y - µ) / σ)
        response.robot_x = 0.0          # Ignored
        response.robot_y = 0.0          #  -"-
        response.robot_radians = 0.0    #  -"-
        return response


class Oracle:

    def __init__(self):
        self.model = ProductionPolicy(x_size=5, g_size=3)
        with open('saved_models/production_policy.pkl', 'rb') as f:
            self.model.load_state_dict(pickle.load(f))

    def sample(self, action_request):
        robot_pose = np.array([[action_request.robot_x, action_request.robot_y, action_request.robot_radians]]).T
        object_pose = np.array([[action_request.object_x, action_request.object_y, action_request.object_radians]]).T
        object_relative_robot = robot_centric(robot_pose, object_pose, translate=True).T
        x = np.concatenate(
            [
                robot_centric(robot_pose, object_pose, translate=True).T,
                [[action_request.object_width, action_request.object_height]]
            ],
            axis=1
        )
        goal_pose = np.array([
            [action_request.object_x_prime, action_request.object_y_prime, action_request.object_radians_prime]
        ]).T
        goal_relative_robot = robot_centric(robot_pose, goal_pose, translate=True).T
        desired_object_change = goal_relative_robot - object_relative_robot
        X = Variable(torch.FloatTensor(x))
        G = Variable(torch.FloatTensor(desired_object_change))

        u = self.model(X, G).data.numpy().T
        action_response = ActionResponse()
        (
            action_response.dx,
            action_response.dy,
            action_response.dr,
            action_response.t
        ) = action_inverse(robot_pose, u).flatten()

        return action_response


def robot_centric(robot_pose, x, translate):
    x = x.copy()
    if translate is True:
        x[:2, :] -= robot_pose[:2, :]
    θ = robot_pose[-1, 0]
    A = np.array([
        [ np.cos(θ), np.sin(θ), 0],
        [-np.sin(θ), np.cos(θ), 0],
        [         0,         0, 1],
    ])
    b = np.array([[0, 0, -θ]]).T
    return A @ x + b


def action_inverse(robot_pose, u):
    θ = robot_pose[-1, 0]
    A = np.array([
        [np.cos(θ),-np.sin(θ), 0, 0],
        [np.sin(θ), np.cos(θ), 0, 0],
        [        0,         0, 1, 0],
        [        0,         0, 0, 1],
    ])
    return A @ u


if __name__ == '__main__':
    from oracle_pb2 import FeasibilityRequest
    oracle = Feasibility()
    request = FeasibilityRequest()
    request.robot_x = 1.25
    request.robot_y = 0.60
    request.robot_radians = np.pi / 2.0

    request.object_x = 1.25
    request.object_y = 0.0
    request.object_radians = 0.0

    request.object_x_prime = 1.25
    request.object_y_prime = 0.1
    request.object_radians_prime = 0.0

    request.object_mass = 0.073
    request.object_rotational_inertia = 0.000064
    request.object_friction = 0.086
    request.object_width = 0.1
    request.object_height = 0.1

    from datetime import datetime
    for _ in range(1):
        start = datetime.now()
        print(oracle.sample(request))
        end = datetime.now()
        #print(end - start)
