import numpy as np
from numpy import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()

        self.add_module('actor', Actor(input_dim, output_dim))
        self.add_module('critic', Critic(input_dim, output_dim))

        self.train()

    def forward(self, x):
        return self.actor(x)


class Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mean = nn.Parameter(torch.zeros(output_dim), requires_grad=True)
        self.covariance = nn.Parameter(torch.ones(output_dim, output_dim))

        self.register_parameter('mean', self.mean)
        self.register_parameter('covariance', self.covariance)

        self.optimizer = torch.optim.Adam(self.parameters())

        self.m = torch.distributions.MultivariateNormal(self.mean, self.covariance)

    def forward(self, x):
        return self.m.rsample()

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.optimizer.step()


class Critic(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.critic_linear = nn.Linear(input_dim + output_dim, 1, bias=True)

        self.register_parameter('critic_linear', self.critic_linear)

        self.train()

    def forward(self, x):
        return self.critic_linear(x)

    def backward(self, loss):
        self.critic_linear.backward()
