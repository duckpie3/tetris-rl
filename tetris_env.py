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
LEFT, RIGHT, DOWN, ROTATE, DROP, HOLD, NONE = 0, 1, 2, 3, 4, 5, 6

# COLORS *********************************************************************

BLACK = (21, 24, 29)
BLUE = (31, 25, 76)
RED = (252, 91, 122)
WHITE = (255, 255, 255)

FPS = 48


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, weights=None, base_fall_interval=24, render_mode: str | None = None):
        super(TetrisEnv, self).__init__()
        self.weights = weights if weights is not None else {
            "aggregate_height": -0.510066,
            "lines_cleared": 0.760666,
            "holes": -0.35663,
            "bumpiness": -0.184483,
        }
        self.base_fall_interval = base_fall_interval
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Dict(
            spaces={
                "piece_type": spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, 0]),
                    high=np.array([1, 1, 1, 1, 1, 1, 1]),
                    shape=(7,),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([0, 0, 0, 0]),
                    high=np.array([1, 1, 1, 1]),
                    shape=(4,),
                    dtype=np.float32,
                ),
                "x": spaces.Box(low=-1, high=COLS - 1, shape=(1,), dtype=np.float32),
                "y": spaces.Box(low=0, high=ROWS - 1, shape=(1,), dtype=np.float32),
                "dist_to_nearest_piece_below": spaces.Box(
                    low=0, high=ROWS, shape=(1,), dtype=np.float32
                ),
                "ticks_to_gravity": spaces.Box(
                    low=0, high=self.base_fall_interval, shape=(1,), dtype=np.float32
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
                "level": spaces.Box(low=1, high=1000, shape=(1,), dtype=np.float32),
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

        rotation_oh_env = np.zeros(4, dtype=np.float32)
        rotation_oh_env[self.tetris.figure.rotation] = 1

        hold_piece_oh_enc = np.zeros(7, dtype=np.float32)
        if self.tetris.hold is not None:
            hold_piece_oh_enc[type_to_num[self.tetris.hold.type]] = 1

        dist_to_nearest_piece_below = 0
        for row in range(self.tetris.figure.y + 4, ROWS):
            for col in range(COLS):
                if self.tetris.board[row][col] != 0:
                    dist_to_nearest_piece_below = row - (self.tetris.figure.y + 4)
                    break
            if dist_to_nearest_piece_below != 0:
                break

        board = np.array(self.tetris.board)
        obs = {
            "piece_type": type_oh_enc,
            "rotation": rotation_oh_env,
            "x": np.array([self.tetris.figure.x], dtype=np.float32),
            "y": np.array([self.tetris.figure.y], dtype=np.float32),
            "dist_to_nearest_piece_below": np.array(
                [dist_to_nearest_piece_below], dtype=np.float32
            ),
            "ticks_to_gravity": np.array(
                [
                    np.clip(
                        self.next_gravity_frame - self.frame, 0, self.base_fall_interval
                    )
                ],
                dtype=np.float32,
            ),
            "next_piece": next_piece_oh_enc,
            "hold_piece": hold_piece_oh_enc,
            "level": np.array([self.tetris.level], dtype=np.float32),
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
        return eval_board(self._get_projected_board(), self.weights)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.tetris = Tetris(ROWS, COLS, seed)

        self.fall_interval = self.base_fall_interval
        self.frame = 0
        self.next_gravity_frame = self.fall_interval
        self.level = self.tetris.level

        self.steps_until_truncated = 40
        self.steps_without_scoring = 0
        obs = self._get_observation()

        info = {}
        return obs, info

    def step(self, action):
        level_p = self.level
        score_before = self.tetris.score

        locked = False
        reward = 0.0
        reward += 0.001 # small living reward to encourage longer games
        phi = self._potential()
        gamma = 0.99
        beta  = 0.5 
        if action == LEFT:
            self.tetris.go_side(-1)
        elif action == RIGHT:
            self.tetris.go_side(1)
        elif action == DOWN:
            locked = self.tetris.go_down()
            reward += 0.01
        elif action == ROTATE:
            self.tetris.rotate()
        elif action == DROP:
            rows_dropped = self.tetris.hard_drop()
            reward += 0.02 * rows_dropped
            locked = True
        elif action == HOLD:
            self.tetris.hold_piece()
        elif action == NONE:
            pass

        if not locked and self.frame >= self.next_gravity_frame:
            locked = self.tetris.go_down()
            self.next_gravity_frame = self.frame + self.fall_interval

        phi_p = self._potential()
        reward += beta * (gamma * phi_p - phi)

        if locked:
            self.level = self.tetris.level
            lines = self.tetris.score - score_before
            reward += 10.0 * lines**2
            self.steps_without_scoring = 0 if lines > 0 else (self.steps_without_scoring + 1)
        terminated = self.tetris.gameover
        truncated = self.steps_without_scoring >= self.steps_until_truncated

        if terminated or truncated:
            reward -= 100.0

        self.frame += 1

        if self.level != level_p and self.level <= 5:
            self.fall_interval = self.base_fall_interval - 4 * (self.level - 1)


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
