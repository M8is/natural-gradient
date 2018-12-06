import torch
from torch import nn
import numpy as np


class ActorCritic(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.actor = Actor(input_dim, output_dim)
        self.critic = Critic(input_dim, output_dim)

        self.train()

    def forward(self, x):
        policy = self.actor(x)
        action = policy.sample()
        return action, policy, self.critic(x, action)


class Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.state_dim = input_dim
        self.action_dim = output_dim

        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2 * self.action_dim)

        self.net = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4
        )

    def forward(self, x):
        net_out = self.net(x)

        loc = net_out[:self.action_dim]
        covariance = torch.diag(net_out[self.action_dim:]) * torch.diag(net_out[self.action_dim:]).t()
        return torch.distributions.MultivariateNormal(loc, covariance)


class Critic(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim + output_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.net = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4
        )

    def forward(self, x, u):
        return self.net(torch.cat((x, u)))
