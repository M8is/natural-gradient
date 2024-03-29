import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

import nac
import quanser_robots
from models.linear_normal import LinearNormal

seed = 36364
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make('Qube-v0')
model = LinearNormal(env.observation_space.shape, env.action_space.shape)

def phi(x):
    A = np.multiply.outer(x, x)
    return np.concatenate((A[np.triu_indices(len(x))], x, np.ones(1)))

gamma = .95
lambda_ = .4
alpha = .2
alpha_decay = .001
h = 10
beta = .001
eps = np.pi / 180
max_episodes = 1000
render = False

try:
    nac.train(env, model, phi, render, gamma, lambda_, alpha, alpha_decay,
              h, beta, eps, max_episodes)
except KeyboardInterrupt:
    print("Interrupted.")

torch.save(model, 'nac_lstd_model.pt')

plt.ion()
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title('Total Returns')
ax1.plot(model.returns)

ax2.set_title('Theta History')
ax2.plot(model.theta_history)

plt.draw()
plt.pause(0.05)

while True:
    done = False
    x = env.reset()
    while not done:
        env.render()
        policy = model(torch.FloatTensor(x))
        u = policy.sample()
        x, r, done, _ = env.step(u.detach().numpy())
