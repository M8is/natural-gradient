import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()

        self.linear = nn.Linear(obs_space[0], act_space[0])
        self.sigma = torch.ones(act_space, requires_grad=True)

    def forward(self, s):
        s = torch.Tensor(s)
        x = self.linear(s)

        N = torch.distributions.normal.Normal(x, self.sigma)

        a = N.sample()
        log_prob = N.log_prob(a)
        return a, log_prob  # torch.normal(x, self.sigma), N.log_prob()


if __name__ == '__main__':
    import gym
    import quanser_robots
    import numpy as np
    from quanser_robots import GentlyTerminating

    env = gym.make('CartpoleStabShort-v0')

    o_space = env.observation_space.shape
    a_space = env.action_space.shape
    #print(o_space)
    #print(a_space)
    l = Linear(o_space, a_space)

    for params in l.parameters():
        params = torch.randn(params.size())
    #x = torch.randn(o_space)
    #print(l(x))

    done = False

    obs = env.reset()
    while not done:
        action = l(obs)
        action = action.detach().numpy()

        print(action)
        obs, _, done, _ = env.step(action)

        env.render()
