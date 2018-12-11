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
        action = policy.sample()
        return action, policy


class Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.distribution = torch.distributions.MultivariateNormal if output_dim > 1 else torch.distributions.Normal

        self.state_dim = input_dim
        self.action_dim = output_dim

        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)

        self._hidden = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3
        )

        self.mean_head = nn.Sequential(
            self._hidden,
            nn.Linear(64, self.action_dim)
        )

        self.cov_head = nn.Sequential(
            self._hidden,
            nn.Linear(64, self.action_dim)
        )

    def forward(self, x):
        loc = self.mean_head(x)
        covariance = self.cov_head(x)
        return self.distribution(loc, covariance)


class Critic(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
