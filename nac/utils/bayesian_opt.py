import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from skopt.optimizer import gp_minimize
from skopt.space import Integer, Real

import quanser_robots
from agents.nacagent import NACAgent, phi
from nac.nac import NACNormal

seed = 36364
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make('CartpoleSwingShort-v0')
model = None
agent = None

try:
    dimensions = [
        Real(0., 1., name="gamma"),
        Real(0., 1., name="lambda"),
        Real(.05, .5, name="alpha"),
        Real(0., .1, name="alpha_decay")
    ]

    def score(params):
        gamma, lambda_, alpha, alpha_decay = params
        env = gym.make('CartpoleSwingShort-v0')
        env_state_dim = env.observation_space.shape[0]
        phi_dim = int(env_state_dim * (env_state_dim + 1) /
                      2) + env_state_dim + 1  # number of quadratic features

        model = NACNormal(env_state_dim, phi_dim, env.action_space.shape[0])
        agent = NACAgent(model, env, gamma=gamma, lambda_=lambda_, alpha=alpha, alpha_decay=alpha_decay)
        model.exception = None

        try:
            agent.train()
        except Exception as e:
            print(repr(e))
            model.exception = repr(e)
        
        torch.save(model, 'models/{}.pt'.format(','.join([str(v) for v in params])))

        return -max(model.total_returns, default=0)

    res = gp_minimize(score, dimensions)

    print("====")
    print(res.x_iters)
    print("====")
    print(res.x)
    print(res.fun)
    print(np.argmin(res.func_vals))
    print("FINISHED")

except KeyboardInterrupt:
    print("Interrupted.")
