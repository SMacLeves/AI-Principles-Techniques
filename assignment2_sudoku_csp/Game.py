import time
class Game:

    def __init__(self, sudoku):
        self.sudoku = sudoku

    def show_sudoku(self):
        print(self.sudoku)

    # MRV heuristic
    def minimum_remaining_values(self, queue, D):
        """
        for the next variable, choose the one with the fewest legal values left in its domain
        so sort the queue based on the domains of first element of every arc.
        """

        sorted_queue = list(sorted(queue, key=lambda arc: len(D[arc[0]])))
        return sorted_queue

    # degree heuristic
    def degree_heuristic(self, queue, D):
        """
        can be applied as a tiebreaker for the mrv
        """

        degree_count = {}
        for arc in queue:
            var = arc[0]
            degree_count[var] = degree_count.get(var, 0) + 1

        degree_sorted_queue = list(sorted(queue, key = lambda arc : degree_count[arc[0]], reverse = True))
        return degree_sorted_queue

    # prioritize finalized arcs heuristic
    def prioritize_finalized_arcs(self, queue, D):
        """
        prioritizes arcs where the second field is finalized
        """

        finalized_sort = list(sorted(queue, key = lambda arc : len(D[arc[1]]) == 1, reverse = True))
        return finalized_sort

    # function to choose which heuristic to apply
    def apply_heuristic(self, queue, D, heuristic = "default"):
        if heuristic == "default":
            return queue  # no sorting
        elif heuristic == "mrv":
            return self.minimum_remaining_values(queue, D)
        elif heuristic == "degree":
            return self.degree_heuristic(queue, D)
        elif heuristic == "finalized":
            return self.prioritize_finalized_arcs(queue, D)
        elif heuristic == "mrv degree":
            mrv_queue = self.minimum_remaining_values(queue, D)
            return self.degree_heuristic(mrv_queue, D)
        elif heuristic == "mrv finalized":
            mrv_queue = self.minimum_remaining_values(queue, D)
            return self.prioritize_finalized_arcs(mrv_queue, D)
        elif heuristic == "degree finalized":
            degree_queue = self.degree_heuristic(queue, D)
            return self.prioritize_finalized_arcs(degree_queue, D)
        elif heuristic == "all":
            mrv_queue = self.minimum_remaining_values(queue, D)
            mrv_degree_queue = self.degree_heuristic(mrv_queue, D)
            return self.prioritize_finalized_arcs(mrv_degree_queue, D)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")

    def solve(self) -> bool:
        """
        Implementation of the AC-3 algorithm
        @return: true if the constraints can be satisfied, false otherwise
        """

        """
        1. list of all arcs (between the given numbers and the dots they constrain)
        2. while list is !empty, remove first arc and ...
        3. REVISE(CSP, Xm, Xn)
            if domain of Xm contains values that conflict with Xn according to problem constraints,
                remove them from Xm, then it is revised
        4. if domain of Xm = 0, return false
        5. if domain of Xm has changed, add to list all arcs (Xk, Xm), where Xk is neighbor of Xm - Xn,
            in order to check for further inconsistencies
        6. if queue is empty and all domains have at least one value -> return true
        """

        def sudoku_to_CSP(sudoku):
            # variables and their domains
            V = []
            D = {}

            # constraints
            C = []

            # fill variables and domains
            for i in range(9):
                for j in range(9):
                    var = f"X{i}{j}"
                    V.append(var)
                    if sudoku.board[i][j].is_finalized():
                        D[var] = [sudoku.board[i][j].get_value()]
                    else:
                        D[var] = list(range(1, 10))

            # add constraints for rows
            for i in range(9):
                row_vars = [f"X{i}{j}" for j in range(9)]
                C.append((row_vars, "all_different"))

            # add constraints for columns
            for j in range(9):
                col_vars = [f"X{i}{j}" for i in range(9)]
                C.append((col_vars, "all_different"))

            # add constraints for 3x3 subgrids
            for box_row in range(3):
                for box_col in range(3):
                    box_vars = []
                    for i in range(3):
                        for j in range(3):
                            var = f"X{3 * box_row + i}{3 * box_col + j}"
                            box_vars.append(var)
                    C.append((box_vars, "all_different"))

            return (V, D, C)

        def create_arcs():
            V, D, C = sudoku_to_CSP(self.sudoku)

            # every element in arcs is an arc, so a tuple of variables
            arcs = []

            """
            access Cs,
            for every constraint,
            access list of vars,
                for every var,
                    if it is finalized,
                        create arcs with all non-finalized
                    else,
                        create arcs with all
            """
            for constraint in C:
                for var in constraint[0]:
                    if len(D[var]) > 1:
                        for xvar in constraint[0]:
                            arc = (var, xvar)
                            if xvar != var and arc not in arcs:
                                arcs.append(arc)

            return arcs, V, D, C

        # runtime measures
        start_time = time.time()
        counter_arcs = 0

        queue, V, D, C = create_arcs()

        # apply heuristic of choice
        heuristic_queue = self.apply_heuristic(queue, D, heuristic = "default")

        while heuristic_queue:
            arc = heuristic_queue.pop(0)
            counter_arcs += 1
            X = arc[0]
            Y = arc[1]
            Xrow = int(X[1])
            Xcol = int(X[2])
            Xdomain = D[X]
            Ydomain = D[Y]

            # this is REVISE()
            if len(Ydomain) == 1:
                value = Ydomain[0]
                if value in Xdomain:
                    Xdomain.remove(value)
                    if len(Xdomain) == 1:
                        self.sudoku.board[Xrow][Xcol].set_value(Xdomain[0])
                    if len(Xdomain) == 0:
                        return False
                    D[X] = Xdomain

                    neighbours = self.sudoku.board[Xrow][Xcol].get_other_neighbours(Y)

                    for neighbour in neighbours:
                        if len(D[neighbour]) > 1:
                            arc = (neighbour, X)
                            if arc not in heuristic_queue:
                                heuristic_queue.append(arc)

                    # apply heuristic of choice
                    heuristic_queue = self.apply_heuristic(heuristic_queue, D, "default")

        end_time = time.time()

        # final progress of trying to solve sudoku
        self.show_sudoku()

        # runtime measures
        print(f"Number of arcs evaluated: {counter_arcs}")
        print(f"Run time: {round(end_time - start_time, 5)} seconds")

        return True



    def valid_solution(self) -> bool:

        """
        Checks the validity of a sudoku solution
        @return: true if the sudoku solution is correct
        """

        goal = list(range(1, 10))
        board = self.sudoku.get_board()

        # check rows
        for row in range(9):
            values = [board[row][col].get_value() for col in range(9)]
            if not sorted(values) == goal:
                return False

        # check columns
        for col in range(9):
            values = [board[row][col].get_value() for row in range(9)]
            if not sorted(values) == goal:
                return False

        # check 3x3 subgrids
        for box_row in range(3):
            for box_col in range(3):
                values = []
                for row in range(box_row * 3, box_row * 3 + 3):
                    for col in range(box_col * 3, box_col * 3 + 3):
                        values.append(board[row][col].get_value())
                if not sorted(values) == goal:
                    return False

        return True
