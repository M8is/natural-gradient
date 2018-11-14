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

        # TODO Compute advantages
        v = []
        for i in range(len(trajectories)):
            vtemp = []
            trajectory = trajectories[i]
            for j in range(len(trajectory['reward'])):
                r = 0
                for t in range(j, len(trajectory['reward'])):
                    r += trajectory['reward'][t]
                vtemp.append(r)
            v.append(vtemp)

        Q = 1  # TODO implement Q-value

        advantages = []
        for i in range(len(trajectories)):
            advantages.append(v[i][0] - Q)
        #print(rewards)
        # print(advantages)

        # TODO Compute Policy Gradient
        pol_grad = 1
        # pol_grad = sum(grads * advantages) / len(trajectories)

        # TODO Compute Fisher
        F = [(grads[i][0] * np.transpose(grads[i][0])) for i in range(len(trajectories))]
        F = sum(F) / len(trajectories)
        # print(F)

        # TODO Gradient Ascent
        # params = params + torch.sqrt(delta / ())

        # TODO Update Parameters of V
