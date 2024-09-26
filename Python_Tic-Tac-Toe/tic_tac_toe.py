"""
Implementation of Tic-Tac-Toe.
"""

import numpy as np

PLAYER_1 = 1
PLAYER_2 = -1

_NUM_ROWS = 3
_NUM_COLS = 3

def _compute_lines_to_check():

	lines = []

	for i in range(_NUM_ROWS):
		lines.append([i*_NUM_COLS + j for j in range(_NUM_COLS)])
		lines.append([j*_NUM_COLS + i for j in range(_NUM_ROWS)])

	lines.append([i*(_NUM_COLS+1) for i in range(_NUM_ROWS)])
	lines.append([(_NUM_COLS-1) * (i+1) for i in range(_NUM_ROWS)])
	return lines
_LINES_TO_CHECK = _compute_lines_to_check()

class GameState:
	def __init__(self):
		self.board = np.zeros(_NUM_ROWS * _NUM_COLS, dtype=int)
		self.legal_moves = list(range(_NUM_ROWS * _NUM_COLS))
		self.current_player = PLAYER_1
		self.winner = 0

	def apply_move(self, move: int) -> None:
		self.board[move] = self.current_player
		self.legal_moves.remove(move)

		# Check if the mover won
		for line in _LINES_TO_CHECK:
			win = True
			for cell in line:
				if self.board[cell] != self.current_player:
					win = False
					break

			if win:
				self.winner = self.current_player
				break

		# Swap players
		self.current_player *= -1

	def undo_move(self, move: int) -> None:

		self.board[move] = 0
		self.legal_moves.append(move)
		self.winner = 0
		self.current_player *= -1

	def is_game_over(self) -> bool:

		return self.winner != 0 or not self.legal_moves

	def print_board(self):
		for i in range(_NUM_ROWS):
			row = self.board[i*_NUM_COLS:(i+1)*_NUM_COLS]
			print(" | ".join(["X" if cell == PLAYER_1 else "O" if cell == PLAYER_2 else " " for cell in row]))
			if i < _NUM_ROWS - 1:
				print("-" * 5)

