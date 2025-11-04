import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tetris import Tetris
from tetris_metrics import eval_board, get_column_height
import pygame
import copy

CELLSIZE = 20
ROWS = 20
COLS = 10
HUD_HEIGHT = 200

WIDTH = COLS * CELLSIZE
HEIGHT = ROWS * CELLSIZE + HUD_HEIGHT
SCREEN = WIDTH, HEIGHT

# COLORS *********************************************************************

BLACK = (21, 24, 29)
BLUE = (31, 25, 76)
RED = (252, 91, 122)
WHITE = (255, 255, 255)

FPS = 8


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, weights=None, render_mode: str | None = None):
        super(TetrisEnv, self).__init__()
        self.weights = weights if weights is not None else {
            "aggregate_height": -0.510066,
            "holes": -0.35663,
            "bumpiness": -0.184483,
            "wells": -0.100000,
        }
        self.render_mode = render_mode
        self.action_space = spaces.MultiDiscrete([2, COLS, 4])
        self.observation_space = spaces.Dict(
            spaces={
                "piece_type": spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, 0]),
                    high=np.array([1, 1, 1, 1, 1, 1, 1]),
                    shape=(7,),
                    dtype=np.float32,
                ),
                "next_piece": spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, 0]),
                    high=np.array([1, 1, 1, 1, 1, 1, 1]),
                    shape=(7,),
                    dtype=np.float32,
                ),
                "hold_piece": spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, 0]),
                    high=np.array([1, 1, 1, 1, 1, 1, 1]),
                    shape=(7,),
                    dtype=np.float32,
                ),
                "column_heights": spaces.Box(
                    low=np.zeros(COLS, dtype=np.float32),
                    high=np.array([ROWS - 1] * COLS, dtype=np.float32),
                    shape=(COLS,),
                    dtype=np.float32,
                ),
            }
        )

    def _get_observation(self):
        if self.tetris.next is None:
            raise ValueError

        type_to_num = {"I": 0, "Z": 1, "S": 2, "J": 3, "L": 4, "T": 5, "O": 6}

        type_oh_enc = np.zeros(7, dtype=np.float32)
        type_oh_enc[type_to_num[self.tetris.figure.type]] = 1

        next_piece_oh_enc = np.zeros(7, dtype=np.float32)
        next_piece_oh_enc[type_to_num[self.tetris.next.type]] = 1

        hold_piece_oh_enc = np.zeros(7, dtype=np.float32)
        if self.tetris.hold is not None:
            hold_piece_oh_enc[type_to_num[self.tetris.hold.type]] = 1

        column_heights = np.array([get_column_height(col, self.tetris.board) for col in range(COLS)], dtype=np.float32)

        obs = {
            "piece_type": type_oh_enc,
            "next_piece": next_piece_oh_enc,
            "hold_piece": hold_piece_oh_enc,
            "column_heights": column_heights,
        }
        return obs
    
    def _potential(self):
        return eval_board(self.tetris.board, self.weights)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.tetris = Tetris(ROWS, COLS, seed)

        self.steps_until_truncated = 40
        self.steps_without_scoring = 0
        obs = self._get_observation()

        info = {}
        if self.render_mode == "human":
            self.tetris.init_render()
        return obs, info

    def step(self, action):
        phi = self._potential()
        gamma = 0.9
        score_before = self.tetris.score

        reward = 0.0
        reward += 0.01 # small living reward to encourage longer games
        hold, x_pos, rotation = action
        if hold == 1:
            if self.tetris.allow_hold:
                self.tetris.hold_piece()
                reward -= 0.1  # small penalty for using hold
            else:
                reward -= 1.0  # larger penalty for invalid hold
        if hold == 0 or (hold == 1 and not self.tetris.allow_hold):
            self.tetris.go_side(x_pos - self.tetris.figure.x)
            for _ in range(rotation):
                self.tetris.rotate()
            self.tetris.hard_drop()
            phi_prime = self._potential()
            reward += gamma * phi_prime - phi
            lines = self.tetris.score - score_before
            reward += 100 * lines**2
            self.steps_without_scoring = 0 if lines > 0 else (self.steps_without_scoring + 1)
        
        terminated = self.tetris.gameover
        truncated = self.steps_without_scoring >= self.steps_until_truncated

        if terminated or truncated:
            reward -= 100.0

        obs = self._get_observation()

        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.tetris.render(FPS)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
            pygame.display.quit()
        return super().close()
