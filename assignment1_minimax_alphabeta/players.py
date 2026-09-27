#Levente Bódi, s1122062
#Konstantinos, Konstantinou s1118596


from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board
import time
import tracemalloc


class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
        Inherits from PlayerController
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth
        self.node_expansion_count = 0
        
    def calculate_branching_factor(self) -> float:
        if self.depth > 0:
            return self.node_expansion_count / self.depth  # Compute the average branching factor
        return 0

    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        #tracemalloc.start()
        self.node_expansion_count = 0  # Reset for each move
        start_time = time.time() # Start measuring time for time complexity
        # Create the root node of the game tree
        root_node = Node(board, self.player_id)
        
        # Generate the tree up to the desired depth
        root_node.generate_children(0)
        
        # Run the minimax algorithm starting at the root
        best_move, best_value = self.minmax(root_node, self.depth, True)
        end_time = time.time()
        branching_factor = self.calculate_branching_factor()
        #current, peak = tracemalloc.get_traced_memory()
        #tracemalloc.stop()
        
        print(f"Minimax run time: {round(end_time-start_time, 3)} seconds")
        print(f"MinMax total nodes expanded: {self.node_expansion_count}")
        print(f"MinMax branching factor: {round(branching_factor, 3)}")
        #print(f"Memory usage in MinMax: {peak / 1024:.2f} KiloBytes")
        print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
        
        return best_move

    def minmax(self, node: Node, depth: int, is_maximizing: bool):
        """Implementing the minimax algorithm

        Args:
            node (Node): the current board state of the game tree
            depth (int): max depth up until the tree should be explored
            is_maximizing (bool): a boolean indicating whether the current level is maximizing or minimizing

        Returns:
            int: the best move
        """
        self.node_expansion_count += 1 # increment the counter for each node expanded
        
        # base case - either the max search depth is reached or the game came to a terminal state
        if depth == 0 or node.board.is_terminal(self.game_n):
            evaluation = self.heuristic.evaluate_board(self.player_id, node.board)
            return None, evaluation
        
        # recursive case of the maximizing player
        if is_maximizing:
            # initializing max_eval and best_move to ensure that the real evaluation score becomes higher than -infinity,
            # and that best_move will store the highest evaluation
            max_eval = float('-inf')
            best_move = None
            # for each child the function recursively calls minimax
            for child in node.children:
                _, eval = self.minmax(child, depth - 1, False)
                # if evaluation from the child node is higher than the current max_eval, max_eval and best_move wil be updated
                if eval > max_eval:
                    max_eval = eval
                    best_move = child.move
                
            return best_move, max_eval
        
        # recursive case of the minimizing player
        # the idea is the same as by the maximizing player, only with minimizing
        else:
            min_eval = float('inf')
            best_move = None
            for child in node.children:
                _, eval = self.minmax(child, depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = child.move
            return best_move, min_eval

class AlphaBetaPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from PlayerController
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth
        self.node_expansion_count = 0
        
    def calculate_branching_factor(self) -> float:
        if self.depth > 0:
            return self.node_expansion_count / self.depth  # Compute the average branching factor
        return 0

    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        #tracemalloc.start()
        self.node_expansion_count = 0  # Reset for each move
        start_time = time.time()
        # Create the root node of the game tree
        root_node = Node(board, self.player_id)
        
        # Generate the tree up to the desired depth
        root_node.generate_children(0)
        
        # Run the minimax algorithm starting at the root
        best_move, best_value = self.alphabeta(root_node, self.depth, float('-inf'), float('inf'), True)
        end_time = time.time()
        branching_factor = self.calculate_branching_factor()
        #current, peak = tracemalloc.get_traced_memory()
        #tracemalloc.stop()
        
        print(f"AlphaBeta run time: {round(end_time-start_time, 3)} seconds")
        print(f"AlphaBeta total nodes expanded: {self.node_expansion_count}")
        print(f"AlphaBeta branching factor: {round(branching_factor, 3)}")
        #print(f"Memory usage in Alpha-Beta: {peak / 1024:.2f} KiloBytes")
        print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
        
        return best_move

    def alphabeta(self, node: Node, depth: int, alpha: float, beta: float, is_maximizing: bool):
        """Implementing the alphabeta pruning algorithm

        Args:
            node (Node): the current board state of the game tree
            depth (int): max depth up until the tree should be explored
            alpha (float): the highest value the maximizing player can guarantee
            beta (float): the lowest value the minimizing player can guarantee
            is_maximizing (bool): a boolean indicating whether the current level is maximizing or minimizing

        Returns:
            int: the best move
        """
        self.node_expansion_count += 1  # increment the counter for each node expanded
        
        if depth == 0 or node.board.is_terminal(self.game_n):
            evaluation = self.heuristic.evaluate_board(self.player_id, node.board)
            return None, evaluation

        if is_maximizing:
            max_eval = float('-inf')
            best_move = None
            for child in node.children:
                # including alpha and beta in the recursive case
                _, eval = self.alphabeta(child, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = child.move
                # updating alpha - threshold subsequent nodes needs to exceed in order to be beneficial
                alpha = max(alpha, eval)
                # pruning decision
                if beta <= alpha:
                    break  # Beta cutoff
            return best_move, max_eval
        # the ideas described in the comments are the same regarding the minimizing player
        else:
            min_eval = float('inf')
            best_move = None
            for child in node.children:
                _, eval = self.alphabeta(child, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = child.move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_move, min_eval


class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
# Implementing the tree structure for the algorithms
class Tree:
    """A tree structure representing the game state
    """

    def __init__(self, root):
        """Initializes the Tree with a root node

        Args:
            root (Node): The root node of the tree
        """
        self.root = root

class Node:
    """Represents a node in the game tree
    """

    def __init__(self, board, player, move=None, parent=None):
        """Initializes a Node with the current game state.

        Args:
            board (Board): the current  board
            player (int): the player who will make the next move
            move (int, optional): the move that led to this node
            parent (Node, optional): the parent node
        """
        self.board = board
        self.player = player
        self.move = move
        self.children = []
        self.parent = parent
        self.evaluation = None

    def generate_children(self, depth, max_depth=6):
        """Recursively generates child nodes up to a maximum depth

        Args:
            depth (int): the current depth in the tree
            max_depth (int, optional): the maximum depth to generate

        Returns:
            None
        """
        valid_moves = []
        if depth >= max_depth:
            return
        for i in range(self.board.width):
            if self.board.is_valid(i):
                valid_moves.append(i)
    
        for move in valid_moves:
            new_board = self.board.get_new_board(move, self.player)
            new_player = 3 - self.player
            child_node = Node(new_board, new_player, move = move, parent=self)
            self.children.append(child_node)
            #Recursive part for generating children with adding 1 depth to the depth value            
            child_node.generate_children(depth + 1, max_depth)







        