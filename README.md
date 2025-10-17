# Tetris RL

A minimal Gymnasium-compatible Tetris environment
for training reinforcement learning agents using Stable Baselines3.
The base Tetris implementation is adapted from
https://github.com/itspyguru/Python-Games.

## Prerequisites

- Python 3.10 or newer
- pip packages: `pygame`, `gymnasium`, `stable-baselines3`, `torch`, `numpy`

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

- `python tetris.py` — play the classic game manually.
  - Controls: arrow keys to move/rotate, space to hard drop, `p` to pause,
    `r` to restart, `q`/`Esc` to quit.
- `python checkenv.py` — validate the Gym environment with the SB3 checker.
- `python train.py` — train a PPO agent; checkpoints are written to
  `models/PPO` and resume automatically if a checkpoint exists.
- `python load.py` — run an existing PPO checkpoint (update `model_path` as
  needed) and render it playing.

Assets used for rendering live under `Assets/` and `Fonts/`. Log output from
training is stored in `logs/` for inspection with TensorBoard.
