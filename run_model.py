import gym
import torch

import quanser_robots
import models

env = gym.make('Qube-v0')
model = torch.load("nac_lstd_model.pt")
agent = None

set_weights_of_iteration = -1
model.set_theta(model.theta_history[set_weights_of_iteration])

while True:
    done = False
    x = env.reset()
    total_return = 0
    while not done:
        env.render()
        policy = model(torch.FloatTensor(x))
        u = policy.sample()
        x, r, done, _ = env.step(u.detach().numpy())
        total_return += r
    print("Return: {:.2E}".format(total_return))
