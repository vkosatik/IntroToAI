import numpy as np

PLAYER_1 = 1
PLAYER_2 = -1

_NUM_ROWS = 3
_NUM_COLS = 3

def _compute_lines_to_check():
    lines = []

    # Rows and Columns
    for i in range(_NUM_ROWS):
        lines.append([i * _NUM_COLS + j for j in range(_NUM_COLS)])  # Rows
        lines.append([j * _NUM_COLS + i for j in range(_NUM_ROWS)])  # Columns

    # Diagonals
    lines.append([i * (_NUM_COLS + 1) for i in range(_NUM_ROWS)])       # Top-left to bottom-right
    lines.append([(_NUM_COLS - 1) * (i + 1) for i in range(_NUM_ROWS)])  # Top-right to bottom-left

    return lines

_LINES_TO_CHECK = _compute_lines_to_check()

class GameState:
    def __init__(self):
        self.board = np.zeros(_NUM_ROWS * _NUM_COLS, dtype=int)
        self.legal_moves = list(range(_NUM_ROWS * _NUM_COLS))
        self.winner = 0

    def apply_move(self, move: int, player: int) -> None:
        self.board[move] = player
        self.legal_moves.remove(move)
        self.check_winner()

    def undo_move(self, move: int) -> None:
        self.board[move] = 0
        self.legal_moves.append(move)
        self.legal_moves.sort()  # Ensure legal_moves is in order
        self.check_winner()

    def check_winner(self):
        self.winner = 0  # Reset winner before checking
        for line in _LINES_TO_CHECK:
            line_values = [self.board[cell] for cell in line]
            if all(value == PLAYER_1 for value in line_values):
                self.winner = PLAYER_1
                break
            elif all(value == PLAYER_2 for value in line_values):
                self.winner = PLAYER_2
                break

    def is_game_over(self) -> bool:
        return self.winner != 0 or not self.legal_moves

    def print_board(self):
        symbols = {PLAYER_1: 'X', PLAYER_2: 'O', 0: ' '}
        for i in range(_NUM_ROWS):
            row = [symbols[self.board[i * _NUM_COLS + j]] for j in range(_NUM_COLS)]
            print(" | ".join(row))
            if i < _NUM_ROWS - 1:
                print("-" * 5)