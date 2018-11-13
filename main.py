from agents.npgagent import NPGAgent
from npg.npg import NPG
from npg.models.linear import Linear

import gym
import quanser_robots

from tqdm import tqdm

env = gym.make('CartpoleStabShort-v0')

policy = NPG(env.observation_space.shape, env.action_space.shape, Linear)
agent = NPGAgent(policy, env)

traj = agent._generate_trajectories(3)
print(traj[0]['action'])
print(traj[1]['action'])
print(traj[2]['action'])
