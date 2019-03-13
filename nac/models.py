import numpy as np
import torch


class LinearNormal(torch.nn.Module):
    def __init__(self, observation_shape: np.ndarray.shape, action_shape: np.ndarray.shape):
        super().__init__()

        self.state_dim = observation_shape[0]
        self.action_dim = action_shape[0]

        self.K = torch.nn.Parameter(.5 * torch.ones(self.action_dim, self.state_dim))
        self.Xi = torch.nn.Parameter(torch.ones(self.action_dim, self.state_dim))

        self.theta_history = []
        self.returns = []

    def theta(self):
        return np.concatenate([param.detach().numpy().flatten() for param in self.parameters()])

    def set_theta(self, new_theta):
        for param in self.parameters():
            values = new_theta[:param.numel()]
            new_theta = new_theta[param.numel():]
            param.data = torch.FloatTensor(values.reshape(param.size()))

    def forward(self, x):
        mean = self.K @ x
        covariance = torch.diag(torch.abs(self.Xi @ x) + np.finfo(float).eps)
        return torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance)
