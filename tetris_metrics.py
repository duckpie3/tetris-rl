def get_blocked_cells(board):
    count = 0
    for col in range(len(board[0])):
        seen_block = False
        for row in range(len(board)):
            if not seen_block:  # Find highest block in col
                if board[row][col] != 0:
                    seen_block = True
            else:  # Count blocked cells
                if board[row][col] == 0:
                    count += 1
    return count


def get_bumpiness(board):
    cols = len(board[0])
    coef = 0
    prev_height = get_column_height(0, board)
    for i in range(1, cols):
        height = get_column_height(i, board)
        coef += abs(height - prev_height)
        prev_height = height
    return coef


def get_max_height(board):
    rows = len(board)
    for r in reversed(range(rows)):
        if all(c == 0 for c in board[r]):
            return rows - r - 1
    return rows - 1

def get_aggregate_height(board):
    total_height = 0
    cols = len(board[0])
    for col in range(cols):
        total_height += get_column_height(col, board)
    return total_height


def get_column_height(col, board):
    rows = len(board)
    for r in range(rows):
        if board[r][col] != 0:
            return rows - r
    return 0
