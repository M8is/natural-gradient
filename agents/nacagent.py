"""Implementation of Natural Actor Critic algorithm."""
import torch

from .baseagent import BaseAgent


class NACAgent(BaseAgent):
    def __init__(self, model, env):
        super().__init__(model, env)

    def train_episode(self, render=False):
        done = False
        obs = self._env.reset()

        rewards = []

        while not done:
            action = self(obs)

            obs, reward, done, _ = self._env.step(action.detach().numpy())

            rewards.append(reward)

            # Gradient ascent



            if render:
                print(reward)
                self._env.render()

        print(sum(rewards) / float(len(rewards)))
