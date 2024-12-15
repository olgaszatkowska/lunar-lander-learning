import gymnasium as gym
from stable_baselines3 import DQN


def get_model(env):
    model = DQN(
        "MlpPolicy",  # Multi-Layer Perceptron policy
        env,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=500,
        verbose=1,
    )
    return model
