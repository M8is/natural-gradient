import gym
from matplotlib import pyplot as plt

import quanser_robots
import torch
from agents.nacagent import NACAgent, phi
from nac.nac import ActorCritic
from skopt.optimizer import gp_minimize
from skopt.space import Real, Integer

env = gym.make('CartpoleSwingShort-v0')
model = None
agent = None

try:
    dimensions = [
        Real(0., 1., name="gamma"),
        Real(0., 1., name="lambda"),
        Real(0., 1., name="alpha"),
        Real(0., 1., name="beta")
    ]

    def score(params):
        gamma, lambda_, alpha, beta = params
        env = gym.make('CartpoleSwingShort-v0')
        env_state_dim = env.observation_space.shape[0]
        phi_dim = int(env_state_dim * (env_state_dim + 1) /
                      2) + env_state_dim + 1  # number of quadratic features

        model = ActorCritic(env_state_dim, phi_dim, env.action_space.shape[0])
        agent = NACAgent(model, env, gamma=gamma, lambda_=lambda_, alpha=alpha, beta=beta)

        try:
            agent.train()
            torch.save(model, 'models/{}.pt'.format('_'.join([str(v) for v in params])))
        except Exception:
            print("EXCEPTION")
            return 0

        return -max(agent.performances)

    res = gp_minimize(score, dimensions)

    print("====")
    print(res.x_iters)
    print("====")
    print(res.x)
    print(res.fun)
    print("FINISHED")

except KeyboardInterrupt:
    print("Interrupted.")
