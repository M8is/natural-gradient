import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from skopt.optimizer import gp_minimize
from skopt.space import Integer, Real

import quanser_robots
import nac

seed = 36364
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make('CartpoleSwingShort-v0')
model = None
agent = None

def phi(x):
    A = np.multiply.outer(x, x)
    return np.concatenate((A[np.triu_indices(len(x))], x, np.ones(1)))

# these values are used for fixed parameters
gamma = .99
lambda_ = 1.
alpha = .1
alpha_decay = .0
h = 1
beta = .0
eps = np.pi / 180
max_episodes: int = 1000

dimensions = [
        Real(0., 1., name="gamma"),
        Real(0., 1., name="lambda"),
        Real(.05, .5, name="alpha"),
        Real(0., .1, name="alpha_decay")
    ]

def score(params):
    gamma, lambda_, alpha, alpha_decay = params
    env = gym.make('Qube-v0')

    model = nac.models.LinearNormal(env.state_dim.shape, env.action_space.shape)
    model.exception = None

    try:
        nac.nac.train(env, model, phi, False, gamma, lambda_, alpha, alpha_decay, h, beta, eps, max_episodes)
    except Exception as e:
        print(repr(e))
        model.exception = repr(e)
    
    torch.save(model, 'models/{}.pt'.format(','.join([str(v) for v in params])))

    return -max(model.total_returns, default=0)
try:
    res = gp_minimize(score, dimensions)
except KeyboardInterrupt:
    print("Interrupted.")

print(res.x)
print(res.fun)
print(np.argmin(res.func_vals))