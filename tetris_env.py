import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tetris import Tetris
from tetris_metrics import eval_board
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

    def __init__(self, weights=None, base_fall_interval=24, render_mode: str | None = None):
        super(TetrisEnv, self).__init__()
        self.weights = weights if weights is not None else {
            "aggregate_height": -0.510066,
            "holes": -0.35663,
            "bumpiness": -0.184483,
        }
        self.base_fall_interval = base_fall_interval
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
                "board": spaces.Box(
                    low=0, high=1, shape=(ROWS * COLS,), dtype=np.float32
                ),
            }
        )
        if self.render_mode == "human":
            pygame.init()
            self.win = pygame.display.set_mode(SCREEN, pygame.NOFRAME)
            self.clock = pygame.time.Clock()
            self.img1 = pygame.image.load("Assets/1.png")
            self.img2 = pygame.image.load("Assets/2.png")
            self.img3 = pygame.image.load("Assets/3.png")
            self.img4 = pygame.image.load("Assets/4.png")
            self.Assets = {1: self.img1, 2: self.img2, 3: self.img3, 4: self.img4}
            self.font = pygame.font.Font("Fonts/Alternity-8w7J.ttf", 50)
            self.font2 = pygame.font.SysFont("cursive", 25)

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

        board = np.array(self.tetris.board)
        obs = {
            "piece_type": type_oh_enc,
            "next_piece": next_piece_oh_enc,
            "hold_piece": hold_piece_oh_enc,
            "board": (board != 0).astype(np.float32).flatten(),
        }
        return obs

    def _get_projected_board(self):
        projection = self.tetris.project_landing()
        board_copy = copy.deepcopy(self.tetris.board)
        for row, col in projection:
            board_copy[row][col] = self.tetris.figure.color
        return board_copy
    
    def _potential(self):
        return eval_board(self.tetris.board, self.weights)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.tetris = Tetris(ROWS, COLS, seed)

        self.steps_until_truncated = 40
        self.steps_without_scoring = 0
        obs = self._get_observation()

        info = {}
        return obs, info

    def step(self, action):
        phi = self._potential()
        gamma = 0.5
        score_before = self.tetris.score

        reward = 0.0
        reward += 0.01 # small living reward to encourage longer games
        hold, x_pos, rotation = action
        if hold == 1:
            self.tetris.hold_piece()

        self.tetris.go_side(x_pos - self.tetris.figure.x)
        for _ in range(rotation):
            self.tetris.rotate()

        self.tetris.hard_drop()
        phi_prime = self._potential()
        reward += gamma * (phi_prime - phi)
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
        tetris = self.tetris
        if self.render_mode == "human":
            self.win.fill(BLACK)
            for x in range(ROWS):
                for y in range(COLS):
                    if tetris.board[x][y] > 0:
                        val = tetris.board[x][y]
                        img = self.Assets[val]
                        self.win.blit(img, (y * CELLSIZE, x * CELLSIZE))
                        pygame.draw.rect(
                            self.win,
                            WHITE,
                            (y * CELLSIZE, x * CELLSIZE, CELLSIZE, CELLSIZE),
                            1,
                        )

            if tetris.figure:
                for i in range(4):
                    for j in range(4):
                        if i * 4 + j in tetris.figure.image():
                            img = self.Assets[tetris.figure.color]
                            x = CELLSIZE * (tetris.figure.x + j)
                            y = CELLSIZE * (tetris.figure.y + i)
                            self.win.blit(img, (x, y))
                            pygame.draw.rect(
                                self.win, WHITE, (x, y, CELLSIZE, CELLSIZE), 1
                            )

            ghost_cells = tetris.project_landing()
            for row, col in ghost_cells:
                ghost_rect = pygame.Rect(
                    col * CELLSIZE, row * CELLSIZE, CELLSIZE, CELLSIZE
                )
                pygame.draw.rect(self.win, WHITE, ghost_rect, 1)

            if tetris.gameover:
                rect = pygame.Rect((50, 140, WIDTH - 100, HEIGHT - 350))
                pygame.draw.rect(self.win, BLACK, rect)
                pygame.draw.rect(self.win, RED, rect, 2)

                over = self.font2.render("Game Over", True, WHITE)

                self.win.blit(over, (rect.centerx - over.get_width() / 2, rect.y + 20))

            # HUD ********************************************************************

            hud_top = HEIGHT - HUD_HEIGHT
            pygame.draw.rect(self.win, BLUE, (0, hud_top, WIDTH, HUD_HEIGHT))
            preview_margin_x = CELLSIZE
            next_origin_y = hud_top + 10
            hold_origin_y = next_origin_y + 4 * CELLSIZE + 20

            if tetris.next:
                next_image = tetris.next.image()
                img = self.Assets[tetris.next.color]
                base_x = preview_margin_x
                for idx in next_image:
                    row, col = divmod(idx, 4)
                    x = base_x + col * CELLSIZE
                    y = next_origin_y + row * CELLSIZE
                    self.win.blit(img, (x, y))

            if tetris.hold:
                hold_image = tetris.hold.image()
                img = self.Assets[tetris.hold.color]
                base_x = preview_margin_x
                for idx in hold_image:
                    row, col = divmod(idx, 4)
                    x = base_x + col * CELLSIZE
                    y = hold_origin_y + row * CELLSIZE
                    self.win.blit(img, (x, y))

            scoreimg = self.font.render(f"{tetris.score}", True, WHITE)
            levelimg = self.font2.render(f"Level : {tetris.level}", True, WHITE)
            self.win.blit(
                scoreimg,
                (WIDTH // 2 - scoreimg.get_width() // 2 + WIDTH // 4, hud_top + 10),
            )
            self.win.blit(
                levelimg,
                (
                    WIDTH // 2 - levelimg.get_width() // 2 + WIDTH // 4,
                    hud_top + HUD_HEIGHT - levelimg.get_height() - 10,
                ),
            )

            pygame.draw.rect(self.win, BLUE, (0, 0, WIDTH, hud_top), 2)
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.update()

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
            pygame.display.quit()
        return super().close()
