import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import Normalization, NormalizationInverse


class FeasibilityGenerator(torch.nn.Module):
    def __init__(self):
        super(FeasibilityGenerator, self).__init__()
        observation_size = 7
        output_size = 4
        self.latent_size = 5
        hidden_size = 512
        input_size = self.latent_size + observation_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.norm2 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.norm3 = torch.nn.BatchNorm1d(hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.norm4 = torch.nn.BatchNorm1d(hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, hidden_size)
        self.fcy = torch.nn.Linear(hidden_size, output_size)
        self.train()

    def forward(self, obj):
        """
        Expects obj: (width, height, x_, y_, θ_) in object frame
        """
        z = Variable(torch.randn(obj.size(0), self.latent_size))
        if obj.is_cuda:
            z = z.cuda()
        inputs = torch.cat([obj, z], dim=1)
        y1 = self.fc1(inputs)
        y2 = self.fc2(F.relu(self.norm2(y1))) + y1
        y3 = self.fc3(F.relu(self.norm3(y2))) + y2
        y4 = self.fc4(F.relu(self.norm4(y3))) + y3
        y = F.tanh(self.fcy(y4))
        return y


class GeneratorProduction(torch.nn.Module):
    def __init__(self):
        super(GeneratorProduction, self).__init__()
        self.generator = FeasibilityGenerator()
        self.obj_norm = Normalization(7)
        self.rob_norminv = NormalizationInverse(4)

    def forward(self, obj, obj_):
        """
        obj       : (width, height, mass, friction, x_obj, y_obj, θ_obj)
        obj_ : (x', y', θ')
        """
        # obj_' is the delta in object frame
        obj_delta = Variable(torch.zeros(obj.size(0), 7))
        x, y, θ = obj.data[0, -3:]
        x_, y_, θ_ = obj_.data[0, -3:]
        θ = obj[:, -1]
        obj_delta[0, :4] = obj[0, :4]
        obj_delta[0, 4] = torch.cos(θ) * (x_ - x) + torch.sin(θ) * (y_ - y)
        obj_delta[0, 5] = -torch.sin(θ) * (x_ - x) + torch.cos(θ) * (y_ - y)
        obj_delta[0, 6] = obj_[0, -1]
        y = self.rob_norminv(self.generator(self.obj_norm(obj_delta)))
        # generator returns: x, y, cos(θ), sin(θ)
        y[:, 2] = torch.atan2(y[:, 3], y[:, 2])
        # invariant: y[:, :3] = (x, y, θ) is now robot position in object frame
        # convert to robot frame
        y_rob = Variable(torch.zeros(obj.size(0), 3))
        y_rob[:, 0] = obj[:, 4] + torch.cos(θ) * y[:, 0] - torch.sin(θ) * y[:, 1]
        y_rob[:, 1] = obj[:, 5] + torch.sin(θ) * y[:, 0] + torch.cos(θ) * y[:, 1]
        θr = y[:, 2] + θ
        y_rob[:, 2] = torch.atan2(torch.sin(θr), torch.cos(θr))
        return y_rob


def to_object_centric(X, Y):
    """
    X : (obj_x, obj_y, obj_θ, width, height, mass, friction)
    Y : (obj_x', obj_y', cos(obj_θ'), sin(obj_θ'))
    returns (width, height, mass, friction, x', y', θ'), all in object frame
    """
    res = torch.zeros(X.size(0), 7)
    if X.is_cuda:
        res = res.cuda()
    res[:, 0] = X[:, 3]  # width
    res[:, 1] = X[:, 4]  # height
    res[:, 2] = X[:, 5]  # mass
    res[:, 3] = X[:, 6]  # friction
    θ = X[:, 2]
    θ_ = torch.atan2(Y[:, 3], Y[:, 2])
    # translate object successor state
    Y_t = Y[:, :2] - X[:, :2]
    res[:, 4] = Y_t[:, 0] * torch.cos(θ) + Y_t[:, 1] * torch.sin(θ)
    res[:, 5] = -Y_t[:, 0] * torch.sin(θ) + Y_t[:, 1] * torch.cos(θ)
    res[:, 6] = torch.atan2(torch.sin(θ_ - θ), torch.cos(θ_ - θ))
    return res


def to_robot_centric(robot, obj):
    """
    Expects (in object centric frame):
    robot : (x_robot, y_robot, θ_robot)
    obj   : (width, height, mass, friction, x', y', θ')

    Returns (in robot frame):
    X : (obj_x, obj_y, obj_θ, width, height, mass, friction)
    Y : (obj_x', obj_y', cos(obj_θ'), sin(obj_θ'))
    """
    X = Variable(torch.zeros(robot.size(0), 7))
    Y = Variable(torch.zeros(robot.size(0), 4))
    if robot.is_cuda:
        X = X.cuda()
        Y = Y.cuda()
    θr = robot[:, 2]
    θo = obj[:, -1]
    X[:, 0] = -robot[:, 0] * torch.cos(θr) - robot[:, 1] * torch.sin(θr)
    X[:, 1] = robot[:, 0] * torch.sin(θr) - robot[:, 1] * torch.cos(θr)
    X[:, 2] = torch.atan2(torch.sin(-θr), torch.cos(-θr))
    X[:, 3] = obj[:, 0]  # width
    X[:, 4] = obj[:, 1]  # height
    X[:, 5] = obj[:, 2]  # mass
    X[:, 6] = obj[:, 3]  # friction
    Y[:, 0] = obj[:, 4] * torch.cos(θr) + obj[:, 5] * torch.sin(θr) + X[:, 0]
    Y[:, 1] = -obj[:, 4] * torch.sin(θr) + obj[:, 5] * torch.cos(θr) + X[:, 1]
    Y[:, 2] = torch.cos(θo - θr)
    Y[:, 3] = torch.sin(θo - θr)
    return X, Y
