import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tetris import Tetris
import pygame

SCREEN = WIDTH, HEIGHT = 300, 500
CELLSIZE = 20
ROWS = (HEIGHT - 120) // CELLSIZE
COLS = WIDTH // CELLSIZE
LEFT, RIGHT, ROTATE, DROP, NONE = 0, 1, 2, 3, 4

# COLORS *********************************************************************

BLACK = (21, 24, 29)
BLUE = (31, 25, 76)
RED = (252, 91, 122)
WHITE = (255, 255, 255)

FPS = 48

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode: str | None = None):
        super(TetrisEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            spaces={
                "piece_type": spaces.Box(low=1, high=7, shape=(1,), dtype=np.float32),
                "rotation":   spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32),
                "x":          spaces.Box(low=-1, high=COLS-1, shape=(1,), dtype=np.float32),
                "y":          spaces.Box(low=0, high=ROWS-1, shape=(1,), dtype=np.float32),
                "next_piece": spaces.Box(low=1, high=7, shape=(1,), dtype=np.float32),
                "board":      spaces.Box(low=0.0, high=4.0, shape=(ROWS * COLS,), dtype=np.float32),
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

    def __del__(self):
        if self.render_mode == "human":
            pygame.quit()

    def _get_observation(self):
        if self.tetris.next is None:
            raise ValueError
        
        type_to_num = {"I": 1, "Z": 2, "S": 3, "J": 4, "L": 5, "T": 6, "O": 7}

        obs = {
            "piece_type": np.array([type_to_num[self.tetris.figure.type]], dtype=np.float32),
            "rotation":   np.array([self.tetris.figure.rotation], dtype=np.float32),
            "x":          np.array([self.tetris.figure.x], dtype=np.float32),
            "y":          np.array([self.tetris.figure.y], dtype=np.float32),
            "next_piece": np.array([type_to_num[self.tetris.next.type]], dtype=np.float32),
            "board":      np.array(self.tetris.board, dtype=np.float32).flatten(),
        }
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.tetris = Tetris(ROWS, COLS)

        self.bumpiness = 0
        self.height = 0
        self.hole_count = 0
        self.score = 0

        self.fall_interval = 24
        self.frame = 0
        self.next_gravity_frame = self.fall_interval
        self.level = self.tetris.level

        obs = self._get_observation()

        info = {}
        return obs, info

    def step(self, action):
        
        bumpiness_p = self.bumpiness
        hole_count_p = self.hole_count
        score_p = self.score

        level_p = self.level

        step_penalty = 0.01
        freezed = False
        reward = 0.0

        if action == LEFT:
            self.tetris.go_side(-1)
        elif action == RIGHT:
            self.tetris.go_side(1)
        elif action == ROTATE:
            self.tetris.rotate()
        elif action == DROP:
            self.tetris.go_space()
            freezed = True
            reward += 0.1
        elif action == NONE:
            step_penalty = 0
        
        if not freezed and self.frame >= self.next_gravity_frame:
            freezed = self.tetris.go_down()
            self.next_gravity_frame += self.fall_interval

        if freezed:
            self.level = self.tetris.level
            self.bumpiness = self.tetris.get_bumpiness()
            self.height = self.tetris.max_height
            self.hole_count = self.tetris.get_hole_count()
            self.score = self.tetris.score
            
            bumpiness_delta = self.bumpiness - bumpiness_p
            hole_count_delta = self.hole_count - hole_count_p
            score_delta = self.score - score_p
            height_penalty = 8 if self.height > 12 else 0

            reward -= bumpiness_delta
            reward -= hole_count_delta * 2
            reward += score_delta**2
            reward -= height_penalty
            reward -= step_penalty

        terminated = self.tetris.gameover
        truncated = False
        if self.tetris.gameover:
            reward -= 15
        
        self.frame += 1

        if self.level - level_p >= 1 and self.level <= 5:
            self.fall_interval -= 4*(self.level-1)

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
                            self.win, WHITE, (y * CELLSIZE, x * CELLSIZE, CELLSIZE, CELLSIZE), 1
                        )

            if tetris.figure:
                for i in range(4):
                    for j in range(4):
                        if i * 4 + j in tetris.figure.image():
                            img = self.Assets[tetris.figure.color]
                            x = CELLSIZE * (tetris.figure.x + j)
                            y = CELLSIZE * (tetris.figure.y + i)
                            self.win.blit(img, (x, y))
                            pygame.draw.rect(self.win, WHITE, (x, y, CELLSIZE, CELLSIZE), 1)

            if tetris.gameover:
                rect = pygame.Rect((50, 140, WIDTH - 100, HEIGHT - 350))
                pygame.draw.rect(self.win, BLACK, rect)
                pygame.draw.rect(self.win, RED, rect, 2)

                over = self.font2.render("Game Over", True, WHITE)

                self.win.blit(over, (rect.centerx - over.get_width() / 2, rect.y + 20))

            # HUD ********************************************************************

            pygame.draw.rect(self.win, BLUE, (0, HEIGHT - 120, WIDTH, 120))
            if tetris.next:
                for i in range(4):
                    for j in range(4):
                        if i * 4 + j in tetris.next.image():
                            img = self.Assets[tetris.next.color]
                            x = CELLSIZE * (tetris.next.x + j - 4)
                            y = HEIGHT - 100 + CELLSIZE * (tetris.next.y + i)
                            self.win.blit(img, (x, y))

            scoreimg = self.font.render(f"{tetris.score}", True, WHITE)
            levelimg = self.font2.render(f"Level : {tetris.level}", True, WHITE)
            self.win.blit(scoreimg, (250 - scoreimg.get_width() // 2, HEIGHT - 110))
            self.win.blit(levelimg, (250 - levelimg.get_width() // 2, HEIGHT - 30))

            pygame.draw.rect(self.win, BLUE, (0, 0, WIDTH, HEIGHT - 120), 2)
            self.clock.tick(FPS)
            pygame.display.update()