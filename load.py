from stable_baselines3 import PPO
from tetris_env import TetrisEnv

models_dir = "models/PPO"

model_path = f"{models_dir}/250000.zip"

env = TetrisEnv(render_mode="human")

model = PPO.load(model_path, env)

num_episodes = 10
for ep in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not(truncated or terminated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

env.close()