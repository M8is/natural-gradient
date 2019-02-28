import gym
from matplotlib import pyplot as plt

import quanser_robots
import torch
from agents.nacagent import NACAgent, phi
from nac.nac import ActorCritic
from skopt.optimizer import gp_minimize
from skopt.space import Real, Integer

env = gym.make('CartpoleSwingShort-v0')
model = torch.load("models/0.23701248769676567_0.9310420868920485_0.1525231855539897_0.3762725506407244.pt")
agent = None

while True:
    done = False
    x = env.reset()
    while not done:
        env.render()
        policy = model(torch.FloatTensor(x))
        u = policy.sample()
        x, r, done, _ = env.step(u.detach().numpy())
