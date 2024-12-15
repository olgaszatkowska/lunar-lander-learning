import gymnasium as gym
from stable_baselines3 import DQN


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def compute_reward(self, state, reward, done):
        """
        Custom reward function that considers the lander's state.
        Args:
            state: The state of the lander (position, velocity, angle, etc.).
            reward: The original reward from the environment.
            done: Whether the episode has ended.
        Returns:
            Modified reward.
        """
        x_pos, y_pos, x_vel, y_vel, angle, angular_vel = state[:6]

        custom_reward = -abs(x_vel) - abs(y_vel) - abs(angle) - abs(angular_vel)

        if done:
            if "landed" in self.env.env.lander.awake:  # Example success condition
                custom_reward += 100
            elif "crashed" in self.env.env.lander.awake:  # Example failure condition
                custom_reward -= 100

        return reward + custom_reward

    def step(self, action):
        state, reward, terminated, truncaked, info = self.env.step(action)
        reward = self.compute_reward(state, reward, terminated)
        return state, reward, terminated, truncaked, info


def get_environment():
    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        gravity=-10.0,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode="human",
    )
    return env
