import pygame
import random

CELLSIZE = 20
ROWS = 20
COLS = 10
HUD_HEIGHT = 200

WIDTH = COLS * CELLSIZE
HEIGHT = ROWS * CELLSIZE + HUD_HEIGHT
SCREEN = WIDTH, HEIGHT
FPS = 24


# COLORS *********************************************************************

BLACK = (21, 24, 29)
BLUE = (31, 25, 76)
RED = (252, 91, 122)
WHITE = (255, 255, 255)


# OBJECTS ********************************************************************


class Tetramino:
    # matrix
    # 0   1   2   3
    # 4   5   6   7
    # 8   9   10  11
    # 12  13  14  15

    FIGURES = {
        "I": [[1, 5, 9, 13], [4, 5, 6, 7]],
        "Z": [[4, 5, 9, 10], [2, 6, 5, 9]],
        "S": [[6, 7, 9, 10], [1, 5, 6, 10]],
        "J": [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        "L": [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        "T": [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        "O": [[1, 2, 5, 6]],
    }

    TYPES = ["I", "Z", "S", "J", "L", "T", "O"]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.choice(self.TYPES)
        self.shape = self.FIGURES[self.type]
        self.color = random.randint(1, 4)
        self.rotation = 0

    def image(self):
        return self.shape[self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.shape)


class Tetris:
    def __init__(self, rows, cols, seed=None):
        self.rows = rows
        self.cols = cols
        self.score = 0
        self.level = 1
        self.board = [[0 for j in range(cols)] for i in range(rows)]
        self.next = None
        self.hold = None
        self.allow_hold = True
        self.gameover = False
        self.max_height = 0
        if seed is not None:
            random.seed(seed)
        self.new_figure()

    def new_figure(self):
        if not self.next:
            self.next = Tetramino(5, 0)
        self.figure = self.next
        self.next = Tetramino(5, 0)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if (
                        i + self.figure.y > self.rows - 1
                        or j + self.figure.x > self.cols - 1
                        or j + self.figure.x < 0
                        or self.board[i + self.figure.y][j + self.figure.x] > 0
                    ):
                        intersection = True
        return intersection

    def remove_line(self):
        rerun = False
        for y in range(self.rows - 1, 0, -1):
            is_full = True
            for x in range(0, self.cols):
                if self.board[y][x] == 0:
                    is_full = False
            if is_full:
                del self.board[y]
                self.board.insert(0, [0 for i in range(self.cols)])
                self.score += 1
                if self.score % 10 == 0:
                    self.level += 1
                rerun = True

        if rerun:
            self.remove_line()

    def freeze(self):
        freezed = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.board[i + self.figure.y][j + self.figure.x] = self.figure.color
                    freezed = True
        self.remove_line()
        self.new_figure()
        if self.intersects():
            self.gameover = True
        self.allow_hold = True
        return freezed

    def project_landing(self):
        if not getattr(self, "figure", None):
            return []

        def collides_at(x, y, image):
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in image:
                        board_y = y + i
                        board_x = x + j
                        if (
                            board_y > self.rows - 1
                            or board_x > self.cols - 1
                            or board_x < 0
                            or self.board[board_y][board_x] > 0
                        ):
                            return True
            return False

        image = self.figure.image()
        ghost_y = self.figure.y
        while not collides_at(self.figure.x, ghost_y + 1, image):
            ghost_y += 1

        projection = []
        for i in range(4):
            for j in range(4):
                if i * 4 + j in image:
                    projection.append((ghost_y + i, self.figure.x + j))
        return projection

    def hold_piece(self):
        if not getattr(self, "figure", None):
            return

        if not self.allow_hold:
            return

        def reset_piece(piece):
            piece.x = self.cols // 2
            piece.y = 0
            piece.rotation = 0
            return piece

        current = self.figure
        if self.hold is None:
            self.hold = reset_piece(current)
            self.new_figure()
        else:
            swap = self.hold
            self.hold = reset_piece(current)
            self.figure = reset_piece(swap)
            self.allow_hold = False

    def hard_drop(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()
            return True
        return False

    def go_side(self, dx):
        self.figure.x += dx
        if self.intersects():
            self.figure.x -= dx

    def rotate(self):
        rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = rotation


def main():
    pygame.init()
    win = pygame.display.set_mode(SCREEN, pygame.NOFRAME)
    clock = pygame.time.Clock()

    # Images
    img1 = pygame.image.load("Assets/1.png")
    img2 = pygame.image.load("Assets/2.png")
    img3 = pygame.image.load("Assets/3.png")
    img4 = pygame.image.load("Assets/4.png")

    Assets = {1: img1, 2: img2, 3: img3, 4: img4}

    # Fonts
    font = pygame.font.Font("Fonts/Alternity-8w7J.ttf", 50)
    font2 = pygame.font.SysFont("cursive", 25)

    counter = 0
    move_down = False
    can_move = True

    tetris = Tetris(ROWS, COLS)

    running = True
    while running:
        win.fill(BLACK)

        counter += 1
        if counter >= 10000:
            counter = 0

        if can_move:
            if counter % (FPS // (tetris.level * 2)) == 0 or move_down:
                if not tetris.gameover:
                    tetris.go_down()

        # EVENT HANDLING *********************************************************
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if can_move and not tetris.gameover:
                    if event.key == pygame.K_LEFT:
                        tetris.go_side(-1)

                    if event.key == pygame.K_RIGHT:
                        tetris.go_side(1)

                    if event.key == pygame.K_UP:
                        tetris.rotate()

                    if event.key == pygame.K_DOWN:
                        move_down = True

                    if event.key == pygame.K_SPACE:
                        tetris.hard_drop()

                if event.key == pygame.K_c:
                    tetris.hold_piece()

                if event.key == pygame.K_r:
                    tetris.__init__(ROWS, COLS)

                if event.key == pygame.K_p:
                    can_move = not can_move

                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    move_down = False

        # tetris.draw_grid()
        for x in range(ROWS):
            for y in range(COLS):
                if tetris.board[x][y] > 0:
                    val = tetris.board[x][y]
                    img = Assets[val]
                    win.blit(img, (y * CELLSIZE, x * CELLSIZE))
                    pygame.draw.rect(
                        win, WHITE, (y * CELLSIZE, x * CELLSIZE, CELLSIZE, CELLSIZE), 1
                    )

        if tetris.figure:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in tetris.figure.image():
                        img = Assets[tetris.figure.color]
                        x = CELLSIZE * (tetris.figure.x + j)
                        y = CELLSIZE * (tetris.figure.y + i)
                        win.blit(img, (x, y))
                        pygame.draw.rect(win, WHITE, (x, y, CELLSIZE, CELLSIZE), 1)

        ghost_cells = tetris.project_landing()
        for row, col in ghost_cells:
            ghost_rect = pygame.Rect(col * CELLSIZE, row * CELLSIZE, CELLSIZE, CELLSIZE)
            pygame.draw.rect(win, WHITE, ghost_rect, 1)

        # GAMEOVER ***************************************************************

        if tetris.gameover:
            rect = pygame.Rect((50, 140, WIDTH - 100, HEIGHT - 350))
            pygame.draw.rect(win, BLACK, rect)
            pygame.draw.rect(win, RED, rect, 2)

            over = font2.render("Game Over", True, WHITE)
            msg1 = font2.render("Press r to restart", True, RED)
            msg2 = font2.render("Press q to quit", True, RED)

            win.blit(over, (rect.centerx - over.get_width() / 2, rect.y + 20))
            win.blit(msg1, (rect.centerx - msg1.get_width() / 2, rect.y + 80))
            win.blit(msg2, (rect.centerx - msg2.get_width() / 2, rect.y + 110))

        # HUD ********************************************************************

        hud_top = HEIGHT - HUD_HEIGHT
        pygame.draw.rect(win, BLUE, (0, hud_top, WIDTH, HUD_HEIGHT))
        preview_margin_x = CELLSIZE
        next_origin_y = hud_top + 10
        hold_origin_y = next_origin_y + 4 * CELLSIZE + 20

        if tetris.next:
            next_image = tetris.next.image()
            img = Assets[tetris.next.color]
            base_x = preview_margin_x
            for idx in next_image:
                row, col = divmod(idx, 4)
                x = base_x + col * CELLSIZE
                y = next_origin_y + row * CELLSIZE
                win.blit(img, (x, y))

        if tetris.hold:
            hold_image = tetris.hold.image()
            img = Assets[tetris.hold.color]
            base_x = preview_margin_x
            for idx in hold_image:
                row, col = divmod(idx, 4)
                x = base_x + col * CELLSIZE
                y = hold_origin_y + row * CELLSIZE
                win.blit(img, (x, y))

        scoreimg = font.render(f"{tetris.score}", True, WHITE)
        levelimg = font2.render(f"Level : {tetris.level}", True, WHITE)
        win.blit(
            scoreimg,
            (WIDTH // 2 - scoreimg.get_width() // 2 + WIDTH // 4, hud_top + 10),
        )
        win.blit(
            levelimg,
            (
                WIDTH // 2 - levelimg.get_width() // 2 + WIDTH // 4,
                hud_top + HUD_HEIGHT - levelimg.get_height() - 10,
            ),
        )

        pygame.draw.rect(win, BLUE, (0, 0, WIDTH, hud_top), 2)
        clock.tick(FPS)
        pygame.display.update()
    pygame.quit()


if __name__ == "__main__":
    main()
