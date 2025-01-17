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
from utils import Normalization, NormalizationInverse


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
        if mahalanobis.data[0] < 1.0:
            y = self.norm_inv(µ + (Y - µ)).data
        else:
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
        from feasibility_gan import GeneratorProduction
        self.generator = GeneratorProduction()
        with open('saved_models/generator_production.pkl', 'rb') as f:
            self.generator.load_state_dict(pickle.load(f))
        self.generator.eval()

    def sample(self, request):
        obj = Variable(torch.FloatTensor([[request.object_width,
                                           request.object_height,
                                           request.object_mass,
                                           request.object_friction,
                                           request.object_x,
                                           request.object_y,
                                           request.object_radians]]))
        obj_ = Variable(torch.FloatTensor([[request.object_x_prime,
                                            request.object_y_prime,
                                            request.object_radians_prime]]))

        x, y, θ = self.generator(obj, obj_).data.numpy().flatten()

        response = FeasibilityResponse()
        response.mahalanobis = -1.0 # Not valid here
        response.robot_x = x
        response.robot_y = y
        response.robot_radians = θ
        return response


class Oracle:

    def __init__(self):
        self.model = ProductionPolicy(x_size=7, g_size=4, u_size=4)
        with open('saved_models/production_policy.pkl', 'rb') as f:
            self.model.load_state_dict(pickle.load(f))

    def sample(self, request):
        # Oracle input arguments:
        # X = (obj_x, obj_y, obj_θ, w, h, m, µ)
        # G = (obj_xd, obj_yd, cos(obj_θd), sin(obj_θd))
        robot_pose = np.array([[request.robot_x, request.robot_y, request.robot_radians]]).T
        object_pose = np.array([[request.object_x, request.object_y, request.object_radians]]).T
        object_relative_robot = robot_centric(robot_pose, object_pose, translate=True).T
        x = np.concatenate(
            [
                object_relative_robot,
                [[request.object_width, request.object_height, request.object_mass, request.object_friction]]
            ],
            axis=1
        )
        goal_pose = np.array([
            [request.object_x_prime, request.object_y_prime, request.object_radians_prime]
        ]).T
        goal_relative_robot = robot_centric(robot_pose, goal_pose, translate=True).T
        X = Variable(torch.FloatTensor(x))
        g = np.array([[
            goal_relative_robot[0, 0], goal_relative_robot[0, 1],
            np.cos(goal_relative_robot[0, 2]), np.sin(goal_relative_robot[0, 2])
        ]])
        G = Variable(torch.FloatTensor(g))

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
    request.robot_x = 0.5
    request.robot_y = 0.0
    request.robot_radians = 0.0

    request.object_x = 0.4
    request.object_y = 0.0
    request.object_radians = 0.0

    request.object_x_prime = 0.5
    request.object_y_prime = 0.0
    request.object_radians_prime = 0.0

    request.object_mass = 0.073
    request.object_rotational_inertia = 0.000064
    request.object_friction = 0.086
    request.object_width = 0.13
    request.object_height = 0.13

    from datetime import datetime
    n_steps = 10
    mcmc_time = 0.0
    gan_time = 0.0

    for n in range(n_steps):
        start = datetime.now()
        print(oracle.sample(request).robot_x)
        end = datetime.now()
        gan_time += (end - start).microseconds / n_steps
    print(gan_time / 1e6)
