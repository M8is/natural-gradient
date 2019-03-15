import sys
import time
from os import makedirs, path

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from skopt.optimizer import gp_minimize
from skopt.space import Integer, Real

import nac
import quanser_robots
from models.linear_normal import LinearNormal

SEED = 9583951
torch.manual_seed(SEED)
np.random.seed(SEED)

MODELS_PATH = 'bo_saved_models_' + time.time()

env = gym.make('Qube-v0')
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
max_episodes = 1000

dimensions = [
        Real(0., 1., name="gamma"),
        Real(0., 1., name="lambda"),
        Real(0., 1., name="alpha"),
        Real(.0, .1, name="alpha_decay"),
        Integer(1, 20, name="h"),
        Real(0., 1., name="beta"),
        Real(np.pi / 720., np.pi / 90., name="eps")
    ]

def score(params):
    gamma, lambda_, alpha, alpha_decay, h, beta, eps = params
    env = gym.make('Qube-v0')

    model = LinearNormal(env.observation_space.shape, env.action_space.shape)
    model.exception = None

    try:
        nac.train(env, model, phi, False, gamma, lambda_, alpha, alpha_decay, h, beta, eps, max_episodes)
    except Exception as e:
        print(repr(e))
        model.exception = repr(e)
    
    torch.save(model, path.join(MODELS_PATH, ','.join([str(v) for v in params]) + '.pt'))

    return -max(model.returns, default=0)

try:
    makedirs(MODELS_PATH)
except OSError:
    sys.exit("Target directory already exists. Aborting...")

res = gp_minimize(score, dimensions)
print(res.x)
print(res.fun)
print(np.argmin(res.func_vals))
