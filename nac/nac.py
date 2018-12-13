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

        self.state_dim = state_dim
        self.action_dim = action_dim

        self._hidden = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 8)
        )
        self.mean_head = torch.nn.Sequential(
            self._hidden,
            torch.nn.Linear(8, self.action_dim)
        )
        self.cov_head = torch.nn.Sequential(
            self._hidden,
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, self.action_dim)
        )

    def theta(self):
        return np.concatenate([param.detach().numpy().flatten() for param in self.parameters()])

    def set_theta(self, new_theta):
        for param in self.parameters():
            values = new_theta[:param.numel()]
            new_theta = new_theta[param.numel():]
            param.data = torch.FloatTensor(values.reshape(param.size()))



    def forward(self, x):
        loc_weights = self.mean_head(x)
        cov_weights = self.cov_head(x)
        covariance_matrix = cov_weights * cov_weights.unsqueeze(-1).t()
        return torch.distributions.MultivariateNormal(loc_weights, covariance_matrix=covariance_matrix)


class Critic(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.weights = np.zeros(state_dim)

    def forward(self, x):
        return x.T * self.weights
