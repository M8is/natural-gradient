import numpy as np
import torch

from .baseagent import BaseAgent


class NACAgent(BaseAgent):
    def __init__(self, model, env, gamma=.99, lambda_=.99, alpha=5e-1, h=10, beta=0., eps=np.pi/180):
        super().__init__(model, env)

        self.beta = beta
        self.h = h
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = eps

        self.theta_deltas = []
        self.performances = []

    def train(self, render=False):
        dim_theta = len(self._model.actor.theta())
        dim_phi = self._model.actor.phi_dim

        w_history = list()

        b = z = np.zeros(dim_phi + dim_theta)
        A = np.zeros((dim_phi + dim_theta, dim_phi + dim_theta))

        phi_x = phi(self._env.reset())

        theta_before = self._model.actor.theta()
        theta_after = theta_before

        i = 0
        t = 0
        done = False
        accu_r = 0
        while True:
            if done:
                self.performances.append(accu_r)
                print(str(i) + ", R: " + str(accu_r))
                accu_r = 0
                i += 1
                phi_x = phi(self._env.reset())

                theta_delta = np.linalg.norm(theta_after - theta_before)
                print("->: " + str(theta_delta))
                self.theta_deltas.append(theta_delta)
                theta_before = self._model.actor.theta()

            if render:
                self._env.render()

            policy = self(torch.FloatTensor(phi_x))
            u = policy.sample()
            log_prob = policy.log_prob(u)

            x1, r, done, _ = self._env.step(u.detach().numpy())
            phi_x1 = phi(x1)

            t += 1
            done = done

            accu_r = self.gamma * accu_r + r

            autograd = torch.autograd.grad(log_prob, self._model.actor.parameters())
            flattened_grads = [grad.numpy().flatten() for grad in autograd]
            grad_theta = np.concatenate(flattened_grads)
            phi_tilde = np.concatenate([phi_x1, np.zeros_like(grad_theta)])
            phi_hat = np.concatenate([phi_x, grad_theta])

            z = self.lambda_ * z + phi_hat
            A = A + np.outer(z, (phi_hat - self.gamma * phi_tilde))
            b = b + z * r

            update = np.linalg.pinv(A) @ b

            w, v = update[:dim_theta], update[dim_theta:]

            self._model.critic.weights = v

            natural_gradient_converged = True
            for tao in range(1, self.h + 1):
                tao += 1
                w_change = angle_between(w, w_history[-tao]) if len(w_history) >= tao else float('+inf')
                natural_gradient_converged = natural_gradient_converged and w_change < self.eps

            if natural_gradient_converged:
                new_theta = self._model.actor.theta() + (self.alpha / (t * 5e-2)) * w
                self._model.actor.set_theta(new_theta)

                z, A, b = self.beta * z, self.beta * A, self.beta * b
                t = 0

                w_history = list()

                theta_after = self._model.actor.theta()

            else:
                w_history.append(w)
                phi_x = phi_x1


def phi(x):
    A = np.outer(x, x)
    return np.concatenate((A[np.triu_indices(len(x))], x, np.ones(1)))


# source: https://stackoverflow.com/a/13849249
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
