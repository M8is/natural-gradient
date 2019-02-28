import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

import quanser_robots
from agents.nacagent import NACAgent, phi
from agents.npgagent import NPGAgent
from nac.nac import ActorCritic
from npg.models.linear import Linear
from npg.npg import NPG

seed = 36364
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make('CartpoleSwingShort-v0')

env_state_dim = env.observation_space.shape[0]
phi_dim = int(env_state_dim * (env_state_dim + 1) /
              2) + env_state_dim + 1  # number of quadratic features

model = ActorCritic(env_state_dim, phi_dim, env.action_space.shape[0])
agent = NACAgent(model, env)

theta_deltas = []
render = False
try:
    agent.train(render=render)
except KeyboardInterrupt:
    print("Interrupted.")

torch.save(model, 'nac_lstd_model.pt')

plt.ion()
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
ax1.set_title('Theta delta')
ax1.plot(agent.theta_deltas)

ax2.set_title('Performance')
ax2.plot(agent.performances)

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
