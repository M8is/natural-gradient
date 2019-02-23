"""Implementation of Natural Actor Critic algorithm using Temporal Difference."""


from .baseagent import BaseAgent
import torch
from torch.distributions.normal import Normal


class TDNACAgent(BaseAgent):
    """NACAgent class."""

    def __init__(self, model, env):
        """Init model and environment."""
        super().__init__(model, env)

    def phi(self, x):
        """TODO: Basis function."""
        return

    def train(self):
        """Implementation skeleton for NAC algorithm."""
        u_space = self._env.env.action_space.shape[0]
        x_space = self._env.env.observation_space.shape[0]

        # Input
        policy = self._model  # mu(u|x) = p(u|x, theta)
        params = list(self._model.model.parameters())  # theta = theta_0
        gradpolicy = torch.autograd.grad(policy, params)  # dlog p(u|x, theta)
        self.phi(self._env.env.xsp)

        # 1: Draw initial state
        advantage = list()
        advantage.append(0)  # A(t+1) = 0

        bias = list()
        bias.append(0)  # b(t+1) = 0

        z = list()
        z.append(0)  # z(t+1) = 0

        u_mean = torch.zeros(u_space)
        x_mean = torch.zeros(x_space)
        u_sigma = torch.ones(u_space, requires_grad=True)
        x_sigma = torch.ones(x_space, requires_grad=True)

        # x(0) ∼ p(x0)
        # should x be drawn from rolling out?
        u = Normal(u_mean, u_sigma).sample()
        x = Normal(x_mean, x_sigma).sample()

        # Discrete time steps
        t_max = 1000
        for t in range(t_max):
            # 3. Execute: Draw action u_t ∼ policy pi(u_t|x_t)
            # x_next ∼ p(x_t+1|x_t, u_t),
            # r = r(x, u)

            # 4. Critic Evaluation
            # assign basis functions
            # set statistics
            # assign critic parameters

            # 5. Actor
            # Update policy parameters
            # Forget statistics
            pass
