import random
import numpy as np
from astropy.units.quantity_helper.function_helpers import place

import tic_tac_toe
from tic_tac_toe import GameState, PLAYER_2, PLAYER_1

def minimax(game_state, depth, is_player1):
    if game_state.is_game_over():
        if game_state.winner == PLAYER_1:
            return 1
        elif game_state.winner == PLAYER_2:
            return -1
        else:
            return 0

    if is_player1:
        best_score = -np.inf
        for move in game_state.legal_moves:
            game_state.apply_move(move)
            score = minimax(game_state, depth + 1, False)
            game_state.undo_move(move)
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = np.inf
        for move in game_state.legal_moves:
            game_state.apply_move(move)
            score = minimax(game_state, depth + 1, True)
            game_state.undo_move(move)
            best_score = min(score, best_score)
        return best_score

def best_move(game_state, is_player1):
    if is_player1:
        best_score = -np.inf
    else:
        best_score = np.inf
    move_to_play = None
    for move in game_state.legal_moves:
        game_state.apply_move(move)
        score = minimax(game_state, 0, not is_player1)
        game_state.undo_move(move)
        if is_player1:
            if score > best_score:
                best_score = score
                move_to_play = move
        else:
            if score < best_score:
                best_score = score
                move_to_play = move
    return move_to_play

if __name__ == "__main__":
    player1_wins = 0
    player2_wins = 0
    draws = 0
    for i in range(100):
        game_state = GameState()
        num_moves = 0
        is_player1_turn = False
        game_state.apply_move(random.choice(game_state.legal_moves))  # to get different final game states
        while not game_state.is_game_over():
            move = best_move(game_state, is_player1_turn)
            game_state.apply_move(move)
            num_moves += 1
            is_player1_turn = not is_player1_turn

        #print(f"Game ended after {num_moves} moves.")
        if game_state.winner == PLAYER_1:
            #print("The winner was: Player 1.")
            player1_wins += 1
        elif game_state.winner == PLAYER_2:
            #print("The winner was: Player 2.")
            player2_wins += 1
        else:
            #print("The game resulted in a draw.")
            draws += 1

        # Print the final game board
        #game_state.print_board()

print(f"Player 1 wins: {player1_wins}")
print(f"Player 2 wins: {player2_wins}")
print(f"Draws: {draws}")
