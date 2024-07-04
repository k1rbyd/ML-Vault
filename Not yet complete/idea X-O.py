import random

# Constants
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
BOARD_SIZE = 3

# Function to print the board
def print_board(board):
    for i in range(BOARD_SIZE):
        print(" | ".join(board[i * BOARD_SIZE:(i + 1) * BOARD_SIZE]))
        if i < BOARD_SIZE - 1:
            print("-" * (BOARD_SIZE * 4 - 1))

# Function to check if there is a winner
def check_winner(board):
    # Check rows
    for i in range(BOARD_SIZE):
        if board[i * BOARD_SIZE] == board[i * BOARD_SIZE + 1] == board[i * BOARD_SIZE + 2] != EMPTY:
            return board[i * BOARD_SIZE]
    # Check columns
    for i in range(BOARD_SIZE):
        if board[i] == board[i + BOARD_SIZE] == board[i + 2 * BOARD_SIZE] != EMPTY:
            return board[i]
    # Check diagonals
    if board[0] == board[4] == board[8] != EMPTY:
        return board[0]
    if board[2] == board[4] == board[6] != EMPTY:
        return board[2]
    # Check for draw
    if all(cell != EMPTY for cell in board):
        return "draw"
    # No winner yet
    return None

# Function to get available moves
def get_available_moves(board):
    return [i for i in range(len(board)) if board[i] == EMPTY]

# Function to make a random move
def random_move(board):
    return random.choice(get_available_moves(board))

# Minimax Algorithm with alpha-beta pruning
def minimax(board, depth, alpha, beta, maximizing_player):
    winner = check_winner(board)
    if winner is not None:
        if winner == "draw":
            return 0
        elif winner == PLAYER_X:
            return 10 - depth
        elif winner == PLAYER_O:
            return depth - 10

    if maximizing_player:
        max_eval = -float('inf')
        for move in get_available_moves(board):
            board[move] = PLAYER_X
            eval = minimax(board, depth + 1, alpha, beta, False)
            board[move] = EMPTY
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_available_moves(board):
            board[move] = PLAYER_O
            eval = minimax(board, depth + 1, alpha, beta, True)
            board[move] = EMPTY
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Function to make a move using Minimax
def minimax_move(board):
    best_move = -1
    best_eval = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    for move in get_available_moves(board):
        board[move] = PLAYER_X
        eval = minimax(board, 0, alpha, beta, False)
        board[move] = EMPTY
        if eval > best_eval:
            best_eval = eval
            best_move = move

    return best_move

# Main function to play the game
def play_game():
    board = [EMPTY] * BOARD_SIZE ** 2
    current_player = PLAYER_X

    while True:
        print_board([str(cell) if cell != EMPTY else " " for cell in board])

        if current_player == PLAYER_X:
            print("Player X's turn:")
            mode = input("Choose mode (1 for random, 2 for Minimax): ")
            if mode == '1':
                move = random_move(board)
            elif mode == '2':
                move = minimax_move(board)
            else:
                print("Invalid mode. Defaulting to random.")
                move = random_move(board)
        else:
            print("Player O's turn:")
            move = int(input("Enter your move (0-8): "))

        if board[move] != EMPTY:
            print("Invalid move. Try again.")
            continue

        board[move] = current_player

        winner = check_winner(board)
        if winner is not None:
            print_board([str(cell) if cell != EMPTY else " " for cell in board])
            if winner == PLAYER_X:
                print("Player X wins!")
            elif winner == PLAYER_O:
                print("Player O wins!")
            else:
                print("It's a draw!")
            break

        current_player *= -1  # Switch player

# Start the game
if __name__ == "__main__":
    play_game()
