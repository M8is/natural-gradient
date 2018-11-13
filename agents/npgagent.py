from .baseagent import BaseAgent


class NPGAgent(BaseAgent):
    def __init__(self, model, env):
        # algorithm = NPG(env.observation_space.shape, env.action_space.shape, model)

        super().__init__(model, env)

    def train_episode(self, num_of_traj):
        trajectories = self._generate_trajectories(num_of_traj)

        # TODO Compute delta log ...

        # TODO Compute advantages

        # TODO Compute Policy Gradient

        # TODO Compute Fisher

        # TODO Update Parameters
