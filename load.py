from pathlib import Path
from stable_baselines3 import PPO
from tetris_env import TetrisEnv
from gymnasium.wrappers import NormalizeObservation, FlattenObservation

def _latest_model_path(models_dir: Path) -> Path:
    """Return the checkpoint with the highest timestep count."""

    checkpoints = []
    for checkpoint in models_dir.glob("*.zip"):
        try:
            timestep = int(checkpoint.stem)
        except ValueError:
            continue
        checkpoints.append((timestep, checkpoint))

    if not checkpoints:
        raise FileNotFoundError(
            f"No valid model checkpoints found in {models_dir}. Train a model first."
        )

    checkpoints.sort(key=lambda item: item[0], reverse=True)
    return checkpoints[0][1]


models_dir = Path("models") / "PPO2"
model_path = _latest_model_path(models_dir)

weights = {
    "aggregate_height": -0.510066,
    "holes": -0.55663,
    "bumpiness": -0.218483,
    "wells": -0.100000,
}

env = NormalizeObservation(FlattenObservation(TetrisEnv(weights=weights, gamma=0.9, beta=1.0, render_mode="human")))

model = PPO.load(str(model_path), env)
print(f"Loaded checkpoint: {model_path}")

num_episodes = 10
for ep in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (truncated or terminated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

env.close()
