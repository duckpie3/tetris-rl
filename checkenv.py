from stable_baselines3.common.env_checker import check_env
from tetris_env import TetrisEnv, ShiftWrapper

env = ShiftWrapper(TetrisEnv())
# It will check your custom environment and output additional warnings if needed
check_env(env)