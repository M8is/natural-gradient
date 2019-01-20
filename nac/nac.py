import numpy as np
import torch


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.train()

    def forward(self, x):
        policy = self.actor(x)
        return policy


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.phi_dim = int(state_dim * (state_dim + 1) / 2) + state_dim + 1  # number of quadratic features
        self.action_dim = action_dim

        self.K = torch.nn.Parameter(torch.zeros(self.action_dim, self.phi_dim))
        # self.Xi = 0.1 * torch.nn.Parameter(torch.ones(self.action_dim, self.phi_dim))

    def theta(self):
        return np.concatenate([param.detach().numpy().flatten() for param in self.parameters()])

    def set_theta(self, new_theta):
        for param in self.parameters():
            values = new_theta[:param.numel()]
            new_theta = new_theta[param.numel():]
            param.data = torch.FloatTensor(values.reshape(param.size()))

    def forward(self, x):
        mean = self.K @ x
        # covariance = torch.diag(0.1 + (1 / 1 + torch.exp(self.Xi @ x)))
        covariance = 0.1 * torch.diag(torch.ones(self.action_dim))
        return torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.weights = np.zeros(state_dim)

    def forward(self, x):
        return x.T * self.weights
