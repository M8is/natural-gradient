import numpy as np
import torch

from .baseagent import BaseAgent


class NACAgent(BaseAgent):
    def __init__(self, model, env, gamma=0.99, lambda_=1e-3, alpha=1e-3, tao=1, beta=0.1):
        super().__init__(model, env)

        self.beta = beta
        self.tao = tao
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

    def train_episode(self, render=False):
        done = False
        x = torch.tensor(self._env.reset())

        w_history = list()

        A = b = 0
        z = torch.zeros(2)

        while not done:
            policy = self(torch.tensor(x))
            u = policy.sample()
            log_prob = policy.log_prob(u)

            x1, r, done, _ = self._env.step(u.detach().numpy())
            phi = torch.tensor(x1, requires_grad=True)

            grad_theta = torch.autograd.grad(log_prob, self._model.actor.theta)[0]
            phi_t = torch.stack([phi, torch.zeros_like(phi)])
            phi_h = torch.stack([phi, grad_theta])

            z = self.lambda_ * z + phi_h
            A = A + z * (phi_h - self.gamma * phi_t).t()
            b = b + z * r

            update = A.inv() * b

            w, v = update

            if angle_between(w, w_history[-self.tao]) < np.finfo(float).eps:
                theta = theta + self.alpha * w
                z = self.beta * z
                A = self.beta * A
                b = self.beta * b

            x = x1

            if render:
                self._env.render()


#  source: https://stackoverflow.com/a/13849249
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))