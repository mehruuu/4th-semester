def solve_n_queens(n):
    """
    Solve the N-Queens problem and return one solution.
    
    Args:
        n: The size of the board and number of queens
        
    Returns:
        A list of column positions for each row, or None if no solution exists
    """
    def is_safe(board, row, col):
        
        
       
        for i in range(col):
            if board[row][i] == 1:
                return False
                
        
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
                
   
        for i, j in zip(range(row, n, 1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
                
        return True
    
    def solve_util(board, col):
       
        if col >= n:
            return True
            
       
        for row in range(n):
            if is_safe(board, row, col):
               
                board[row][col] = 1
                
            
                if solve_util(board, col + 1):
                    return True
                    
                
                board[row][col] = 0
             
        return False
    
   
    board = [[0 for _ in range(n)] for _ in range(n)]
    
    if not solve_util(board, 0):
        return None 
    
    
    solution = []
    for row in range(n):
        for col in range(n):
            if board[row][col] == 1:
                solution.append(col)
                break
    
    return solution

def print_board(solution, n):
    """Print the chessboard with queens placed according to solution."""
    for row in range(n):
        line = ""
        for col in range(n):
            if solution[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()


n = 8  
solution = solve_n_queens(n)
if solution:
    print(f"Solution for {n}-Queens problem:")
    print_board(solution, n)
else:
    print(f"No solution exists for {n}-Queens problem.")