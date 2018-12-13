import numpy as np
import torch

from .baseagent import BaseAgent


class NACAgent(BaseAgent):
    def __init__(self, model, env, gamma=0.99, lambda_=0.4, alpha=0.99, tao=1, beta=1e-50, eps=1e-5):
        super().__init__(model, env)

        self.beta = beta
        self.tao = tao
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = eps

        self.theta_deltas = []

    def train(self, render=False):
        dim_theta = len(self._model.actor.theta())
        dim_phi = self._model.actor.state_dim
        w_history = list()

        b = z = np.zeros(dim_phi + dim_theta)
        A = np.zeros((dim_phi + dim_theta, dim_phi + dim_theta))

        x0 = self._env.reset()
        phi = x0

        theta_before = self._model.actor.theta()
        theta_delta = float('+inf')

        while True or theta_delta > self.eps:
            policy = self(torch.tensor(phi))
            u = policy.sample()
            log_prob = policy.log_prob(u)

            x1, r, done, _ = self._env.step(u.detach().numpy())
            phi1 = x1

            flattened_grads = [grad.numpy().flatten() for grad in torch.autograd.grad(log_prob, self._model.actor.parameters())]
            grad_theta = np.concatenate(flattened_grads)
            phi_tilde = np.concatenate([phi1, np.zeros_like(grad_theta)])
            phi_hat = np.concatenate([phi, grad_theta])

            z = self.lambda_ * z + phi_hat
            A = A + z * (phi_hat - self.gamma * phi_tilde).T
            b = b + z * r

            update = np.linalg.pinv(A) @ b

            w, v = update[:dim_theta], update[-dim_phi:]

            self._model.critic.weights = v
            w_change = angle_between(w, w_history[-self.tao]) if len(w_history) >= self.tao else float('+inf')
            if w_change < self.eps:
                new_theta = self._model.actor.theta() + self.alpha * w
                self._model.actor.set_theta(new_theta)

                z = self.beta * z
                A = self.beta * A
                b = self.beta * b

                w_history = list()

                theta_after = self._model.actor.theta()
                theta_delta = np.linalg.norm(theta_after - theta_before)
                self.theta_deltas.append(theta_delta)
                print(theta_delta)

                x0 = self._env.reset()
                phi1 = x0

                theta_before = self._model.actor.theta()
            else:
                w_history.append(w)

            phi = phi1

            if render:
                self._env.render()


# source: https://stackoverflow.com/a/13849249
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))