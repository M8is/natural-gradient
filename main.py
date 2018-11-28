from agents.npgagent import NPGAgent
from agents.nacagent import NACAgent
from npg.npg import NPG
from nac.nac import ActorCritic
from npg.models.linear import Linear

import gym
import quanser_robots

from tqdm import tqdm

env = gym.make('CartpoleStabShort-v0')

#policy = NPG(env.observation_space.shape, env.action_space.shape, Linear)
#agent = NPGAgent(policy, env)

#agent.train_episode(3)


model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
agent = NACAgent(model, env)

for i in range(10000):
    agent.train_episode(render=False)