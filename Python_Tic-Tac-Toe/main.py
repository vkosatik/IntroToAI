import random
import numpy as np
import time
import psutil
from tic_tac_toe import GameState, PLAYER_1, PLAYER_2

def minimax(game_state, player):
    if game_state.is_game_over():
        if game_state.winner == PLAYER_1:
            return 1  # Player 1 wins
        elif game_state.winner == PLAYER_2:
            return -1  # Player 2 wins
        else:
            return 0  # Draw

    if player == PLAYER_1:
        best_score = -np.inf
        for move in game_state.legal_moves:
            game_state.apply_move(move, player)
            score = minimax(game_state, PLAYER_2)
            game_state.undo_move(move)
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = np.inf
        for move in game_state.legal_moves:
            game_state.apply_move(move, player)
            score = minimax(game_state, PLAYER_1)
            game_state.undo_move(move)
            best_score = min(score, best_score)
        return best_score

def best_move(game_state, player):
    if player == PLAYER_1:
        best_score = -np.inf
    else:
        best_score = np.inf
    move_to_play = None
    for move in game_state.legal_moves:
        game_state.apply_move(move, player)
        score = minimax(game_state, -player)
        game_state.undo_move(move)
        if player == PLAYER_1:
            if score > best_score:
                best_score = score
                move_to_play = move
        else:
            if score < best_score:
                best_score = score
                move_to_play = move
    return move_to_play

if __name__ == "__main__":

    start_time = time.perf_counter()
    process = psutil.Process()


    player1_wins = 0
    player2_wins = 0
    draws = 0
    total_memory_usage = 0

    for i in range(50):
        game_state = GameState()
        num_moves = 0
        current_player = PLAYER_1

        random_move = random.choice(game_state.legal_moves)
        game_state.apply_move(random_move, current_player)
        num_moves += 1
        current_player = -current_player

        while not game_state.is_game_over():
            move = best_move(game_state, current_player)
            game_state.apply_move(move, current_player)
            num_moves += 1
            current_player = -current_player

        print(f"Game {i+1} ended after {num_moves} moves.")
        if game_state.winner == PLAYER_1:
            player1_wins += 1
        elif game_state.winner == PLAYER_2:
            player2_wins += 1
        else:
            draws += 1

        game_state.print_board()
        print()

        memory_after_each_game = process.memory_info().rss / (1024 ** 2)
        total_memory_usage += memory_after_each_game

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Player 1 wins: {player1_wins}")
    print(f"Player 2 wins: {player2_wins}")
    print(f"Draws: {draws}")
    print(f'Total memory used over 100 trials: {total_memory_usage:.2f} MB')