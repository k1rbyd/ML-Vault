import numpy as np
import random

class SudokuSolver:
    def __init__(self):
        self.Q = {}  # Q-table for state-action values
        self.epsilon = 0.1  # Exploration rate
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.9  # Discount factor

    def train(self, num_episodes=30000):
        for episode in range(num_episodes):
            self.__train_episode()

    def __train_episode(self):
        sudoku_grid = self.__generate_random_sudoku()

        state = self.__state_to_string(sudoku_grid)
        while not self.__is_solved(sudoku_grid):
            action = self.__select_move(state)
            if action is None:
                break
            next_state, reward = self.__play_move(state, action)
            self.__update_Q(state, action, reward, next_state)
            state = next_state

    def solve_interactively(self):
        print("Enter the Sudoku puzzle row by row. Use '0' for empty cells.")
        sudoku_grid = []
        for i in range(9):
            row = list(map(int, input(f"Enter row {i+1} (9 numbers separated by spaces): ").strip().split()))
            sudoku_grid.append(row)
        
        print("\nSudoku to solve:")
        self._print_sudoku(sudoku_grid)
        
        solved_grid = self.solve(sudoku_grid)
        
        print("\nSolved Sudoku:")
        self._print_sudoku(solved_grid)

    def solve(self, sudoku_grid):
        state = self.__state_to_string(sudoku_grid)
        while not self.__is_solved(sudoku_grid):
            action = self.__select_move(state)
            if action is None:
                break
            sudoku_grid = self.__apply_action(sudoku_grid, action)
            state = self.__state_to_string(sudoku_grid)
        return sudoku_grid

    def __generate_random_sudoku(self):
        base = np.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                base[i, j] = random.randint(1, 9) if random.random() < 0.5 else 0
        return base.tolist()

    def __update_Q(self, state, action, reward, next_state):
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = 0.0
        
        best_next_action = max(self.Q[next_state], key=self.Q[next_state].get) if next_state in self.Q else None
        td_target = reward + self.gamma * (self.Q[next_state][best_next_action] if best_next_action is not None else 0.0)
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta

    def __state_to_string(self, state):
        return ''.join(str(num) if num != 0 else '0' for row in state for num in row)

    def __string_to_state(self, state_str):
        state = []
        for i in range(0, len(state_str), 9):
            row = [int(state_str[i+j]) for j in range(9)]
            state.append(row)
        return state

    def __get_allowed_moves(self, state):
        allowed_moves = []
        for i in range(9):
            for j in range(9):
                if state[i][j] == 0:
                    for num in range(1, 10):
                        if self.__is_valid_move(state, i, j, num):
                            allowed_moves.append((i, j, num))
        return allowed_moves

    def __select_move(self, state):
        if random.random() < self.epsilon or state not in self.Q:
            allowed_moves = self.__get_allowed_moves(self.__string_to_state(state))
            return random.choice(allowed_moves) if allowed_moves else None
        else:
            return max(self.Q[state], key=self.Q[state].get, default=None)

    def __play_move(self, state, action):
        if action is None:
            return state, 0  # No valid move found, return current state and no reward
        
        i, j, num = action
        next_state = self.__apply_action(self.__string_to_state(state), action)
        reward = 1 if self.__is_solved(next_state) else 0
        next_state_str = self.__state_to_string(next_state)
        return next_state_str, reward

    def __apply_action(self, state, action):
        i, j, num = action
        state[i][j] = num
        return state

    def __is_solved(self, state):
        for i in range(9):
            for j in range(9):
                if state[i][j] == 0 or not self.__is_valid_move(state, i, j, state[i][j]):
                    return False
        return True

    def __is_valid_move(self, state, row, col, num):
        # Check row and column constraints
        for i in range(9):
            if state[row][i] == num or state[i][col] == num:
                return False
        
        # Check 3x3 grid constraints
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if state[i][j] == num:
                    return False
        
        return True

    def _print_sudoku(self, state):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print('- - - - - - - - - - - - ')
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print(' | ', end='')
                if j == 8:
                    print(state[i][j])
                else:
                    print(state[i][j], end=' ')

if __name__ == '__main__':
    solver = SudokuSolver()
    print("Training the Sudoku solver...")
    solver.train(1000)
    print("Training completed.")

    while True:
        solver.solve_interactively()
        
        play_again = input("\nDo you want to solve another Sudoku puzzle? (yes/no): ").lower()
        if play_again != 'yes':
            break

    print("Thanks for solving Sudoku!")
