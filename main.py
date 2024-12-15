from model import get_model
from environment import get_environment, RewardWrapper


env = RewardWrapper(get_environment())
model = get_model(env)

print("Training the DQN agent...")
model.learn(total_timesteps=1)
print("Training completed!")

model.save("lunarlander_dqn")
print("Model saved as 'lunarlander_dqn'")

print("Testing the trained DQN agent...")
vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1):
    action, _ = model.predict(obs, deterministic=True)
    state, reward, terminated, truncaked = vec_env.step(action)
    vec_env.render()
    if terminated:
        obs = vec_env.reset()

vec_env.close()
print("Testing completed!")
