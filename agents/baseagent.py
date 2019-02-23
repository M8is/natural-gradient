class BaseAgent:
    def __init__(self, model, env):
        self._model = model
        self._env = env

    def __call__(self, observation, log_prob=False):
        return self._model(observation, log_prob) if log_prob else self._model(observation)

    def _generate_trajectory(self, render=False):
        done = False
        traj = {'observation': [],
                'action': [],
                'reward': [],
                'log_prob': []}

        obs = self._env.reset()
        while not done:
            action, log_prob = self(obs, True)

            traj['observation'].append(obs)
            traj['action'].append(action)
            traj['log_prob'].append(log_prob)

            obs, reward, done, _ = self._env.step(action.detach().numpy())

            traj['reward'].append(reward)

            if render:
                print(reward)
                self._env.render()

        return traj

    def _generate_trajectories(self, n):
        return [self._generate_trajectory() for _ in range(n)]
