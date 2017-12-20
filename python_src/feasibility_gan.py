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
        Expects obj: (width, height, dx, dy, dθ)
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

    def forward(self, obj, obj_delta):
        """
        obj       : (width, height, mass, friction, x_obj, y_obj, θ_obj)
        obj_delta : (dx, dy, dθ)
        """
        # obj_delta' is the delta in object frame
        obj_delta_ = Variable(torch.zeros(obj.size(0), 7))
        θ = obj[:, -1]
        obj_delta_[:, :4] = obj[:, :4]
        obj_delta_[:, 4] = torch.cos(θ) * obj_delta[:, 0] + torch.sin(θ) * obj_delta[:, 1]
        obj_delta_[:, 5] = -torch.sin(θ) * obj_delta[:, 0] + torch.cos(θ) * obj_delta[:, 1]
        obj_delta_[:, 6] = obj_delta[:, -1]
        y = self.rob_norminv(self.generator(self.obj_norm(obj_delta_)))
        y[:, 2] = torch.atan2(y[:, 3], y[:, 2])
        # invariant: y is now robot position in object frame
        # convert to robot frame
        y_rob = Variable(torch.zeros(obj.size(0), 3))
        y_rob[:, 0] = obj[:, 4] + torch.cos(θ) * y[:, 0] - torch.sin(θ) * y[:, 1]
        y_rob[:, 1] = obj[:, 5] + torch.sin(θ) * y[:, 0] + torch.cos(θ) * y[:, 1]
        θr = y[:, 2] + θ
        y_rob[:, 2] = torch.atan2(torch.sin(θr), torch.cos(θr))
        return y_rob

