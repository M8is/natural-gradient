class BaseAgent:
    def __init__(self, algorithm, env):
        self._algorithm = algorithm
        self._env = env

    def __call__(self, observation):
        return self._algorithm(observation)

    def _generate_trajectory(self, render=False):
        done = False
        traj = {'observation': [],
                'action': [],
                'reward': []}

        obs = self._env.reset()
        while not done:
            action = self(obs)
            action = action.detach().numpy()

            traj['observation'].append(obs)
            traj['action'].append(action)

            obs, reward, done, _ = self._env.step(action)

            traj['reward'].append(reward)

            if render:
                print(reward)
                self._env.render()

        return traj

    def _generate_trajectories(self, n):
        return [self._generate_trajectory() for _ in range(n)]
