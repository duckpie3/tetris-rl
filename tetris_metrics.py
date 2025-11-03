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

def get_wells(board):
    wells = 0.0  # Float for fractional scoring if desired
    cols = len(board[0])
    if cols < 3:
        return 0  # Can't have wells in <3 cols
    
    heights = [get_column_height(c, board) for c in range(cols)]  # Precompute for efficiency
    
    for col in range(1, cols - 1):  # Internal columns only
        col_height = heights[col]
        left_h = heights[col - 1]
        right_h = heights[col + 1]
        
        # Well depth: how much lower than the lower neighbor
        if left_h > col_height and right_h > col_height:
            depth = min(left_h, right_h) - col_height
            wells += depth  # Or max(0, depth - threshold) for deeper bonus
    
    return wells

def clear_lines(board):
    new_board = [row for row in board if any(cell == 0 for cell in row)]
    lines_cleared = len(board) - len(new_board)
    for _ in range(lines_cleared):
        new_board.insert(0, [0] * len(board[0]))
    return new_board, lines_cleared

def eval_board(board, weights=None):
    if weights is None:
        weights = {
            "aggregate_height": -0.510066,
            "holes": -0.55663,
            "bumpiness": -0.184483,
            "wells": -0.100000,
        }

    aggregate_height = get_aggregate_height(board)
    holes = get_blocked_cells(board)
    bumpiness = get_bumpiness(board)
    wells = get_wells(board)

    score = (
        weights["aggregate_height"] * aggregate_height +
        weights["holes"] * holes +
        weights["bumpiness"] * bumpiness +
        weights["wells"] * wells
    )

    return score