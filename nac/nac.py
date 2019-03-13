from typing import Callable

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from nac.utils.vector_utils import angle_between


def train(env: gym.Env, model: torch.nn.Module, phi: Callable[np.ndarray, np.ndarray], render: bool, 
          gamma: float, lambda_: float, alpha: float, alpha_decay: float, h: int, 
          beta: float, eps: float, max_episodes: int):
    env_state_dim = env.observation_space.shape

    dim_theta = model.theta().shape
    dim_phi = phi(np.zeros_like(env_state_dim)).shape

    w_history = list()

    b = z = np.zeros(dim_phi + dim_theta)
    A = np.zeros((dim_phi + dim_theta, dim_phi + dim_theta))

    x = env.reset()
    phi_x = phi(x)

    theta_before = model.theta()
    new_theta = None

    i = 0
    epochs = 0
    done = False
    discounted_return = 0
    total_return = 0
    gamma = 1
    while i < max_episodes:
        if done:
            model.returns.append(total_return)

            print(str(i) + ",dR: " + "{:.2E}".format(discounted_return))
            print(str(i) + ", R: " + "{:.2E}".format(total_return))

            theta_delta = np.linalg.norm(new_theta - theta_before)
            if theta_delta:
                print("-> " + "{:.2E}".format(theta_delta))
            theta_before = new_theta

            discounted_return = 0
            total_return = 0
            i += 1
            x = env.reset()
            phi_x = phi(x)

        if render:
            env.render()

        policy = model(torch.FloatTensor(x))
        u = policy.sample()
        log_prob = policy.log_prob(u)

        x1, r, done, _ = env.step(u.detach().numpy())
        phi_x1 = phi(x1)

        discounted_return += gamma * r
        gamma *= gamma

        total_return += r

        autograd = torch.autograd.grad(log_prob, model.parameters())
        grad_theta = np.concatenate([grad.numpy().flatten() for grad in autograd])
        phi_tilde = np.concatenate([phi_x1, np.zeros_like(grad_theta)])
        phi_hat = np.concatenate([phi_x, grad_theta])

        z = lambda_ * z + phi_hat
        A = A + np.multiply.outer(z, (phi_hat - gamma * phi_tilde))
        b = b + z * r

        try:
            if not np.linalg.matrix_rank(A) == len(b):
                update = np.linalg.lstsq(A, b, rcond=1e-3)[0]
            else:
                update = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        w = update[:dim_theta]

        if len(w_history) < h:
            natural_gradient_converged = False
        else:
            natural_gradient_converged = True
            for tao in range(1, h + 1):
                angle_converged = angle_between(w, w_history[-tao]) < eps
                approx_same = np.linalg.norm(w - w_history[-tao]) < np.finfo(float).eps
                natural_gradient_converged = (angle_converged or approx_same) and natural_gradient_converged

        w_history.append(w)

        if natural_gradient_converged:
            learning_rate = alpha / (alpha_decay * epochs + 1)
            new_theta = model.theta() + learning_rate * w
            model.set_theta(new_theta)
            model.theta_history.append(new_theta)

            z, A, b = beta * z, beta * A, beta * b
            w_history = list()
            epochs += 1
            gamma = 1


        x = x1
        phi_x = phi_x1
