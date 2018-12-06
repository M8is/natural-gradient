import numpy as np
import torch

from .baseagent import BaseAgent


class NACAgent(BaseAgent):
    def __init__(self, model, env, discount_rate=0.99, learning_rate=1e-3):
        super().__init__(model, env)

        self.policy_optimizer = torch.optim.Adam(list(model.actor.parameters()), lr=1e-3)
        self.value_optimizer = torch.optim.Adam(list(model.critic.parameters()), lr=5e-3)
        self.discount_rate = discount_rate
        self.lr = learning_rate

        self.loss_function = torch.nn.MSELoss()

    def train_episode(self, render=False, max_length=200):
        done = False
        obs = self._env.reset()

        actions = []
        critic_values = []
        actor_losses = []
        accumulated_rewards = []

        for _ in range(max_length):
            if done:
                break

            action, policy, critic_value = self(torch.tensor(obs))
            critic_values.append(critic_value)

            actions.append(action)

            obs, reward, done, _ = self._env.step(action.detach().numpy())
            reward = torch.from_numpy(np.array(reward))
            accumulated_reward = accumulated_rewards[-1] * self.discount_rate + reward if accumulated_rewards else reward

            actor_losses.append(-policy.log_prob(action) * critic_value)
            accumulated_rewards.append(accumulated_reward)

            if render:
                self._env.render()

        accumulated_rewards = torch.stack(accumulated_rewards)
        critic_values = torch.cat(critic_values)
        value_loss = self.loss_function(critic_values, accumulated_rewards)
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        policy_loss = torch.stack(actor_losses).sum()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # advantage = accumulated_rewards - critic_values
        # for param in self._model.actor.parameters():
        #   param.data = param.data + self.lr * advantage * param.data.grad

        return policy_loss, value_loss
