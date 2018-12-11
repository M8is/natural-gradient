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

actor_losses = []
critic_losses = []
render = False
try:
    for i in range(1000000):
        print('.', end='', flush=True)

        actor_loss, critic_loss = agent.train_episode(render=render)
        actor_losses.append(actor_loss.detach().numpy())
        critic_losses.append(critic_loss.detach().numpy())

        if i > 0 and not i % 100:
            print()
            print(str(i) + ', actor: ' + str(actor_loss.item()) + ', critic: ' + str(critic_loss.item()))
            render = True
        else:
            render = False
except KeyboardInterrupt:
    print("Interrupted.")

f, (ax1, ax2) = plt.subplots(2, 1, sharex='all')

ax1.set_title('Actor loss')
ax1.plot(actor_losses)
ax2.set_title('Critic loss')
ax2.plot(critic_losses)

plt.show()
