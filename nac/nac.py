from typing import Callable

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.vector_utils import angle_between


def train(env: gym.Env, model: torch.nn.Module, phi: Callable[[np.ndarray], np.ndarray], render: bool, 
          gamma: float, lambda_: float, alpha: float, alpha_decay: float, h: int, 
          beta: float, eps: float, max_episodes: int):
    """
        TODO: docstring
    """

    dim_theta = model.theta().shape[0]
    dim_phi = phi(np.zeros(model.state_dim)).shape[0]

    b = z = np.zeros(dim_phi + dim_theta)
    A = np.zeros((dim_phi + dim_theta, dim_phi + dim_theta))

    x = env.reset()
    phi_x = phi(x)

    episodes = 0
    epochs = 0
    done = False
    total_return = 0
    w_history = list()
    while episodes < max_episodes:
        if done:
            model.returns.append(total_return)
            print(str(episodes) + ", R: " + "{:.2E}".format(total_return))
            total_return = 0
            episodes += 1
            x = env.reset()
            phi_x = phi(x)
        if render:
            env.render()

        policy = model(torch.FloatTensor(x))
        u = policy.sample()
        log_prob = policy.log_prob(u)

        x1, r, done, _ = env.step(u.detach().numpy())
        phi_x1 = phi(x1)
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
                update = np.linalg.lstsq(A, b, rcond=None)[0]
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
            old_theta = model.theta()
            learning_rate = alpha / (alpha_decay * epochs + 1)
            new_theta = model.theta() + learning_rate * w
            model.set_theta(new_theta)
            model.theta_history.append(new_theta)

            theta_delta = np.linalg.norm(new_theta - old_theta)
            print("-> " + "{:.2E}".format(theta_delta))

            z, A, b = beta * z, beta * A, beta * b
            w_history = list()
            epochs += 1

        x = x1
        phi_x = phi_x1
