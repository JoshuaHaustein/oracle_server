import torch
import numpy as np
from torch.autograd import Variable


class DropoutExtended(torch.nn.Module):
    """Dropout layer for Dropout VI networks

    Masks are sampled during both training and evaluation.
    Supports (un)freezing masks. For batch inputs, the
    same mask applies to all rows.

    Parameters
    ----------
    p : float, optional
        dropout probability, default=0.05
    """
    def __init__(self, p=0.05):
        super(DropoutExtended, self).__init__()
        self.p = p
        self.mask = None
        self.frozen = False

    def freeze(self):
        self.mask = None
        self.frozen = True

    def unfreeze(self):
        self.mask = None
        self.frozen = False

    def forward(self, x):
        if not self.frozen or self.mask is None:
            self.mask = Variable(
                torch.bernoulli(torch.ones(*x.size()) * (1 - self.p))
            )
            if x.is_cuda:
                self.mask = self.mask.cuda()
        if x.size(0) != self.mask.size(0):
            raise ValueError('Batch size must be constant for a frozen dropout layer!')
        y = x * self.mask
        if not self.frozen:
            self.mask = None
        return y


class Normalization(torch.nn.Module):
    """Normalizes input by running mean and variance. Running
    mean is updated during training."""
    def __init__(self, n_features, gamma=0.99):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.zeros(1, n_features))
        self.register_buffer('var', torch.ones(1, n_features))
        self.γ = gamma

    def _update_running(self, x):
        k = x.size(0)
        x_ = x.data
        self.mean = self.mean * self.γ ** k + (1 - self.γ ** k) * x_.mean(dim=0)
        self.var = self.var * self.γ ** k + (1 - self.γ ** k) * ((x_ - self.mean.repeat(k, 1)) ** 2).mean(dim=0)

    def forward(self, x):
        if self.training:
            self._update_running(x)
        k = x.size(0)
        return (x - Variable(self.mean.repeat(k, 1))) / torch.sqrt(Variable(self.var).repeat(k, 1))


class CircleNorm(torch.nn.Module):
    """Normalizes two consecutive elements of a vector by the euclidean norm

    Parameters
    ----------
    start_indexes : list
        first index of each pair
    """
    def __init__(self, start_indexes):
        super(CircleNorm, self).__init__()
        assert type(start_indexes) is list
        self.start_indexes = start_indexes

    def forward(self, x):
        y = Variable(torch.zeros(x.size()))
        y[:, :] = x
        for i in self.start_indexes:
            y[:, i:i + 2] = x[:, i:i + 2] / torch.norm(x[:, i:i + 2], dim=1).repeat(1, 2)
        return y


class HSDBBALoss:
    """Heteroscedastic Dropout BB-α loss

    Parameters
    ----------
    alpha : float, optional
        alpha-divergence parameter, default=0.5
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __call__(self, mus, sigmas, y):
        """
        Parameters
        ----------
        mus : [torch.autograd.Variable]
            list of sampled predicted means
        sigmas : [torch.autograd.Variable]
            list of sampled predicted standard deviations, not
            variances!
        y : torch.autograd.Variable
            target variables
        """
        α = self.alpha

        logits = Variable(torch.zeros(y.size(0), len(mus)))

        if mus[0].is_cuda:
            logits = Variable(torch.zeros(y.size(0), len(mus))).cuda()

        for k, (µ, σ) in enumerate(zip(mus, sigmas)):
            logits[:, k] = -α * (
                torch.log(σ).sum(dim=1) +
                0.5 * torch.norm((µ - y) / σ, dim=1) ** 2
            )

        max_logits = logits.max(dim=1, keepdim=True)[0]
        sum_exp = torch.exp(logits - max_logits).sum(dim=1)

        return -1 / α * (max_logits + torch.log(sum_exp)).sum()


def angle_expand(x, i):
    """Expands angle, at index i, to cos(angle) and sin(angle)

    Example:
    >>> angle_expand(np.array([0.5, np.pi / 4], 0.0), 1)
    array([ 0.5       ,  0.70710678,  0.70710678,  0.        ])
    """
    y = np.zeros(x.shape[0] + 1)
    y[:i] = x[:i]
    y[i] = np.cos(x[i])
    y[i + 1] = np.sin(x[i])
    y[i + 2:] = x[i + 1:]
    return y


def angles_expand(x, inds):
    """Expands angle, at indices `inds`, to cos(angle) and sin(angle)
    """
    y = x
    for i, ind in enumerate(sorted(inds)):
        y = angle_expand(y, ind + i)
    return y
