import pickle

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from oracle_pb2 import (
    ActionResponse,
    FeasibilityResponse,
    FeasibilitySampleResponse,
    PushabilityResponse
)


hidden_size = 256 # Same for all models


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
        pass

    def mahalanobis(self, pushability_request):
        response = PushabilityResponse()
        response.mahalanobis = np.linalg.norm([
            pushability_request.object_relative_x_prime / 2.0,
            pushability_request.object_relative_y_prime / 2.0,
        ])
        return response


class Feasibility:

    def __init__(self):
        self.x_size = 6
        self.y_size = 4
        hidden_size = 256
        self.mean = torch.nn.Sequential(
            torch.nn.Linear(self.x_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.y_size),
        ).cuda()
        self.mean.eval()
        with open('saved_models/feasibility_mean_model.pkl', 'rb') as f:
            self.mean.load_state_dict(pickle.load(f))

        self.std = torch.nn.Sequential(
            torch.nn.Linear(self.x_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.y_size),
            torch.nn.Softplus()
        ).cuda()
        self.std.eval()
        with open('saved_models/feasibility_std_model.pkl', 'rb') as f:
            self.std.load_state_dict(pickle.load(f))

        # Normalization consants
        self.µx = Variable(torch.cuda.FloatTensor([[
            0.003809,
            0.003643,
           -0.002864,
            0.017553,
            0.344222,
           -0.001551,
        ]]))
        self.σx = Variable(torch.cuda.FloatTensor([[
            0.708319,
            0.705899,
            2.017727,
            2.025916,
            0.737853,
            0.580616,
        ]]))
        self.µy = Variable(torch.cuda.FloatTensor([[
            0.006034,
           -0.008209,
           -0.001866,
            0.001664,
        ]]))
        self.σy = Variable(torch.cuda.FloatTensor([[
            1.096717,
            1.098555,
            0.704748,
            0.709479,
        ]]))

    def sample(self, sample_request):
        X = (Variable(torch.cuda.FloatTensor([[
            np.cos(sample_request.object_radians),
            np.sin(sample_request.object_radians),
            sample_request.object_relative_x_prime,
            sample_request.object_relative_y_prime,
            np.cos(sample_request.object_radians_prime),
            np.sin(sample_request.object_radians_prime),
        ]])) - self.µx) / self.σx
        µ = self.mean(X)
        σ = self.std(X)
        z = Variable(torch.randn(µ.size())).cuda()
        y = ((z * σ + µ) * self.σy + self.µy).cpu().data
        response = FeasibilitySampleResponse()
        response.robot_relative_x = y[0, 0]
        response.robot_relative_y = y[0, 1]
        response.robot_radians = np.arctan2(y[0, 2], y[0, 3])
        return response

    def mahalanobis(self, feasibility_request):
        X = (Variable(torch.cuda.FloatTensor([[
            np.cos(feasibility_request.object_radians),
            np.sin(feasibility_request.object_radians),
            feasibility_request.object_relative_x_prime,
            feasibility_request.object_relative_y_prime,
            np.cos(feasibility_request.object_radians_prime),
            np.sin(feasibility_request.object_radians_prime),
        ]])) - self.µx) / self.σx
        Y = (Variable(torch.cuda.FloatTensor([[
            feasibility_request.robot_relative_x,
            feasibility_request.robot_relative_y,
            np.cos(feasibility_request.robot_radians),
            np.sin(feasibility_request.robot_radians),
        ]])) - self.µy) / self.σy
        µ = self.mean(X)
        σ = self.std(X)
        distance = torch.norm((Y - µ) / σ)
        response = FeasibilityResponse()
        response.mahalanobis = distance.cpu().data[0]
        return response


class Oracle:

    def __init__(self):
        self.x_size = oracle_x_size = 10
        oracle_y_size = 5
        oracle_n_residual_units = 2

        self.mean = torch.nn.Sequential(
            torch.nn.Linear(oracle_x_size, hidden_size),
            *[Residual(hidden_size) for _ in range(oracle_n_residual_units)],
            torch.nn.Linear(hidden_size, oracle_y_size)
        ).cuda()
        self.mean.eval()
        with open('saved_models/oracle_mean_model.pkl', 'rb') as f:
            self.mean.load_state_dict(pickle.load(f))

        self.std = torch.nn.Sequential(
            torch.nn.Linear(oracle_x_size, hidden_size),
            *[Residual(hidden_size) for _ in range(oracle_n_residual_units)],
            torch.nn.Linear(hidden_size, oracle_y_size),
            torch.nn.Softplus()
        ).cuda()
        self.std.eval()
        with open('saved_models/oracle_std_model.pkl', 'rb') as f:
            self.std.load_state_dict(pickle.load(f))

        # Normalization consants
        self.µx = Variable(torch.cuda.FloatTensor([[
             0.006034,
            -0.008209,
            -0.001866,
             0.001664,
             0.003809,
             0.003643,
            -0.002864,
             0.017553,
             0.344222,
            -0.001551,
        ]]))
        self.σx = Variable(torch.cuda.FloatTensor([[
            1.096717,
            1.098555,
            0.704748,
            0.709479,
            0.708319,
            0.705899,
            2.017727,
            2.025916,
            0.737853,
            0.580616,
        ]]))
        self.µy = Variable(torch.cuda.FloatTensor([[
            -0.001666,
             0.016130,
             0.004708,
             0.567908,
             5.037682,
        ]]))
        self.σy = Variable(torch.cuda.FloatTensor([[
            1.205601,
            1.208147,
            0.903146,
            0.257854,
            3.801545,
        ]]))


    def sample(self, action_request):
        X = (Variable(torch.cuda.FloatTensor([[
            action_request.robot_relative_x,
            action_request.robot_relative_y,
            np.cos(action_request.robot_radians),
            np.sin(action_request.robot_radians),
            np.cos(action_request.object_radians),
            np.sin(action_request.object_radians),
            action_request.object_relative_x_prime,
            action_request.object_relative_y_prime,
            np.cos(action_request.object_radians_prime),
            np.sin(action_request.object_radians_prime),
        ]])) - self.µx) / self.σx
        µ = self.mean(X)
        σ = self.std(X)
        z = Variable(torch.randn(µ.size())).cuda()
        y = ((z * σ + µ) * self.σy + self.µy).cpu().data
        action_response = ActionResponse()
        action_response.dx = y[0, 0]
        action_response.dy = y[0, 1]
        action_response.dr = y[0, 2]
        action_response.t = y[0, 3]
        return action_response
