import torch
from torch import nn
import numpy as np


class ActorCritic(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.actor = Actor(input_dim, output_dim)
        self.critic = Critic(input_dim)

        self.train()

    def forward(self, x):
        policy = self.actor(x)
        return policy


class Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.distribution = torch.distributions.MultivariateNormal if output_dim > 1 else torch.distributions.Normal

        self.state_dim = input_dim
        self.action_dim = output_dim

        self.theta = torch.stack([torch.zeros(self.action_dim, requires_grad=True), torch.ones(self.action_dim, requires_grad=True)])

    def forward(self, x):

        return self.distribution(self.theta[0], torch.diag(self.theta[1]))


class Critic(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.weights = torch.zeros(input_dim)

    def forward(self, x):
        return x.t() * self.weights
