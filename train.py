from stable_baselines3 import PPO
from tetris_env import TetrisEnv
from pathlib import Path

models_dir = Path("models") / "PPO"
logs_dir = Path("logs")

models_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

env = TetrisEnv()

latest_checkpoint = None
latest_timestep = 0
checkpoints = sorted(models_dir.glob("*.zip"), key=lambda p: int(p.stem), reverse=True)
if checkpoints:
    latest_checkpoint = checkpoints[0]
    latest_timestep = int(latest_checkpoint.stem)

if latest_checkpoint:
    print(f"Loading existing model from {latest_checkpoint}")
    model = PPO.load(str(latest_checkpoint), env=env, tensorboard_log=str(logs_dir))
else:
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=str(logs_dir))

TIMESTEPS = 10_000
start_iteration = latest_timestep // TIMESTEPS
for i in range(start_iteration, start_iteration + 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(str(models_dir / f"{TIMESTEPS * (i + 1)}"))
