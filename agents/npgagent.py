from .baseagent import BaseAgent
import torch
import numpy as np


class NPGAgent(BaseAgent):
    def __init__(self, model, env):
        # algorithm = NPG(env.observation_space.shape, env.action_space.shape, model)

        super().__init__(model, env)

    def train_episode(self, num_of_traj):
        # Collect trajectories
        trajectories = self._generate_trajectories(num_of_traj)
        a = trajectories[0]['action']
        o = trajectories[0]['observation']
        r = trajectories[0]['reward']
        l = trajectories[0]['log_prob']

        params = list(self._model.model.parameters())

        # Compute Grad(log ....) [step 4.]
        grads = []
        for trajectory in trajectories:
            grads.append([])
            for i in range(len(trajectory['action'])):
                grads[-1].append(torch.autograd.grad(trajectory['log_prob'][i], params, retain_graph=True, create_graph=True))
        # print(grads)

        # TODO Compute advantages
        rewards = []
        for i in range(len(trajectories)):
            rewards.append(np.sum(trajectories[i]['reward']))

        advantages = []
        for i in range(len(trajectories)):
            advantages.append(np.mean(trajectories[i]['reward']))
        #print(rewards)
        #print(advantages)

        # TODO Compute Policy Gradient
        # pol_grad =

        # TODO Compute Fisher

        # TODO Update Parameters
