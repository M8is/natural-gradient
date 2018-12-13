import torch

from agents.npgagent import NPGAgent
from agents.nacagent import NACAgent
from npg.npg import NPG
from nac.nac import ActorCritic
from npg.models.linear import Linear

import gym
import quanser_robots

from matplotlib import pyplot as plt

env = gym.make('CartpoleStabShort-v0')

#policy = NPG(env.observation_space.shape, env.action_space.shape, Linear)
#agent = NPGAgent(policy, env)

#agent.train_episode(3)


model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
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

plt.draw()
plt.pause(0.05)

while True:
    done = False
    x = env.reset()
    while not done:
        env.render()
        policy = model(torch.tensor(x))
        u = policy.sample()
        x, r, done, _ = env.step(u.detach().numpy())