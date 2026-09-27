# AI Principles & Techniques — Programming Assignments

Three programming assignments for the **AI Principles & Techniques** course at Radboud University (Nov 2024), covering classic AI search, constraint satisfaction, and probabilistic inference.

Authors: Levente Bódi (s1122062), Konstantinos Konstantinou (s1118596)

## Contents

| Folder | Topic | Report |
|---|---|---|
| [`assignment1_minimax_alphabeta/`](assignment1_minimax_alphabeta) | Minimax & alpha-beta pruning for "N in a row" | [report.pdf](assignment1_minimax_alphabeta/report.pdf) |
| [`assignment2_sudoku_csp/`](assignment2_sudoku_csp) | Sudoku as a CSP, solved with AC-3 + heuristics | [report.pdf](assignment2_sudoku_csp/report.pdf) |
| [`assignment3_bayesnet_variable_elimination/`](assignment3_bayesnet_variable_elimination) | Exact inference in Bayesian networks via Variable Elimination | [report.pdf](assignment3_bayesnet_variable_elimination/report.pdf) |

### 1. Minimax & Alpha-Beta Pruning (`assignment1_minimax_alphabeta/`)

Implements the **Minimax** algorithm and its **alpha-beta pruning** optimization for a generalized "N in a row" (Connect-Four-style) game, and benchmarks them across board sizes, search depths, and win-condition lengths.

- `board.py` — board representation and win/terminal-state checks
- `heuristics.py` — heuristic evaluation functions for non-terminal states
- `players.py` — `HumanPlayer`, `MinMaxPlayer`, `AlphaBetaPlayer` controllers
- `app.py` — entry point; configures the board size, `N`, and the two players, then runs the game

Run with:
```bash
cd assignment1_minimax_alphabeta
python app.py
```
Findings: alpha-beta pruning explores orders of magnitude fewer nodes than plain minimax (e.g. ~2,900 vs ~1.95M nodes at N=4, an 11x10 board, depth 10), with the biggest gains at greater search depth and board size.

### 2. Sudoku as a CSP — AC-3 (`assignment2_sudoku_csp/`)

Models Sudoku as a **Constraint Satisfaction Problem** (81 variables, domains {1..9}, row/column/subgrid uniqueness constraints) and solves it by enforcing arc consistency with **AC-3**, experimenting with variable-ordering heuristics (MRV, degree heuristic, priority-to-finalized-variables, and combinations).

- `Sudoku.py` — puzzle representation loaded from a text file
- `Field.py` — a single Sudoku cell/variable and its domain
- `Game.py` — CSP setup, the AC-3 revise/propagate loop, and the heuristics
- `App.py` — interactive entry point that lets you pick and solve one of the sample puzzles
- `Sudokus/` — five sample puzzles (`Sudoku1.txt` … `Sudoku5.txt`)

Run with:
```bash
cd assignment2_sudoku_csp
python App.py
```
Then enter a puzzle number (1-5) when prompted. Note: AC-3 alone solves puzzles 1, 2, and 5; puzzles 3 and 4 need search/backtracking beyond arc consistency (as discussed in the report).

### 3. Bayesian Networks — Variable Elimination (`assignment3_bayesnet_variable_elimination/`)

Implements the general **Variable Elimination** algorithm for exact inference over Bayesian networks described in `.bif` format, with a "contained-in-fewest-factors-first" elimination-order heuristic.

- `read_bayesnet.py` — parses `.bif` network files into nodes/parents/probability tables
- `variable_elim.py` — the Variable Elimination algorithm (factor construction, multiplication, summing-out, normalization)
- `run.py` — entry point; loads a network, sets a query/observed variables, and prints the resulting distribution
- `earthquake.bif`, `earthquake1.bif`, `survey.bif` — sample Bayesian networks (`.bif` format, e.g. from the [bnlearn repository](http://www.bnlearn.com/bnrepository/))

Run with:
```bash
cd assignment3_bayesnet_variable_elimination
python run.py
```
This runs the example query `P(Alarm | Burglary=True)` on `earthquake1.bif` and also writes a step-by-step log to `variable_elimination_log.txt`.

## Requirements

- Python 3.10+
- See [`requirements.txt`](requirements.txt): `numpy`, `numba` (assignment 1), `pandas` (assignment 3)
- Assignment 2 uses only the Python standard library

```bash
pip install -r requirements.txt
```

## License

MIT — see [LICENSE](LICENSE).
