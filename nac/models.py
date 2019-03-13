import numpy as np
import torch


class LinearNormal(torch.nn.Module):
    def __init__(self, observation_shape: np.ndarray.shape, action_shape: np.ndarray.shape):
        super().__init__()

        self.state_dim = observation_shape[0]
        self.action_dim = action_shape[0]

        self.loc_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.action_dim)
        )

        self.cov_net = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, self.action_dim),
            torch.nn.Softplus()
        )

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
        mean = self.loc_net(x)
        cov = torch.diag(self.cov_net(x))
        return torch.distributions.MultivariateNormal(mean, scale_tril=cov)
