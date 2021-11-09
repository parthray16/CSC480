# Name:         Parth Ray
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Board Stupid
# Term:         Summer 2021

import math
import random
from typing import Callable, Generator, Optional, Tuple, List, Dict


class GameState:

    def __init__(self, board: Tuple[Tuple[Optional[int], ...], ...],
                 player: int) -> None:
        """
        An instance of GameState has the following attributes.

            player: Set as either 1 (MAX) or -1 (MIN).
            moves: A tuple of integers representing empty indices of the board.
            selected: The index that the current player believes to be their
                      optimal move; defaults to -1.
            util: The utility of the board; either 1 (MAX wins), -1 (MIN wins),
                  0 (tie game), or None (non-terminal game state).
            traverse: A callable that takes an integer as its only argument to
                      be used as the index to apply a move on the board,
                      returning a new GameState object with this move applied.
                      This callable provides a means to traverse the game tree
                      without modifying parent states.
            display: A string representation of the board, which should only be
                     used for debugging and not parsed for strategy.

        In addition, instances of GameState may be stored in hashed
        collections, such as sets or dictionaries.

        >>> board = ((   0,    0,    0,    0,   \
                         0,    0, None, None,   \
                         0, None,    0, None,   \
                         0, None, None,    0),) \
                    + ((None,) * 16,) * 3

        >>> state = GameState(board, 1)
        >>> state.util
        None
        >>> state.player
        1
        >>> state.moves
        (0, 1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(0)
        >>> state.player
        -1
        >>> state.moves
        (1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(5)
        >>> state.player
        1
        >>> state.moves
        (1, 2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(1)
        >>> state.player
        -1
        >>> state.moves
        (2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(10)
        >>> state.player
        1
        >>> state.moves
        (2, 3, 4, 8, 12, 15)
        >>> state = state.traverse(2)
        >>> state.player
        -1
        >>> state.moves
        (3, 4, 8, 12, 15)
        >>> state = state.traverse(15)
        >>> state.player
        1
        >>> state.moves
        (3, 4, 8, 12)
        >>> state = state.traverse(3)
        >>> state.util
        1
        """
        self.player: int = player
        self.moves: Tuple[int] = GameState._get_moves(board, len(board))
        self.selected: int = -1
        self.util: Optional[int] = GameState._get_utility(board, len(board))
        self.traverse: Callable[[int], GameState] = \
            lambda index: GameState._traverse(board, len(board), player, index)
        self.display: str = GameState._to_string(board, len(board))
        self.keys: Tuple[int, ...] = tuple(hash(single) for single in board)

    def __eq__(self, other: "GameState") -> bool:
        return self.keys == other.keys

    def __hash__(self) -> int:
        return hash(self.keys)

    @staticmethod
    def _traverse(board: Tuple[Tuple[Optional[int], ...], ...],
                  width: int, player: int, index: int) -> "GameState":
        """
        Return a GameState instance in which the board is updated at the given
        index by the current player.

        Do not call this method directly; instead, call the |traverse| instance
        attribute, which only requires an index as an argument.
        """
        i, j = index // width ** 2, index % width ** 2
        single = board[i][:j] + (player,) + board[i][j + 1:]
        return GameState(board[:i] + (single,) + board[i + 1:], -player)

    @staticmethod
    def _get_moves(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> Tuple[int]:
        """
        Return a tuple of the unoccupied indices remaining on the board.
        """
        return tuple(j + i * width ** 2 for i, single in enumerate(board)
                     for j, square in enumerate(single) if square == 0)

    @staticmethod
    def _get_utility(board: Tuple[Tuple[Optional[int], ...], ...],
                     width: int) -> Optional[int]:
        """
        Return the utility of the board; either 1 (MAX wins), -1 (MIN wins),
        0 (tie game), or None (non-terminal game state).
        """
        for line in GameState._iter_lines(board, width):
            if line == (1,) * width:
                return 1
            if line == (-1,) * width:
                return -1
        return 0 if len(GameState._get_moves(board, width)) == 0 else None

    @staticmethod
    def _iter_lines(board: Tuple[Tuple[Optional[int], ...], ...],
                    width: int) -> Generator[Tuple[int], None, None]:
        """
        Iterate over all groups of indices that represent a winning condition.
        X lines are row-wise, Y lines are column-wise, and Z lines go through
        all single boards; combinations of these axes refer to the direction
        of the line in 2D or 3D space.
        """
        for single in board:
            # x lines (2D rows)
            for i in range(0, len(single), width):
                yield single[i:i + width]
            # y lines (2D columns)
            for i in range(width):
                yield single[i::width]
            # xy lines (2D diagonals)
            yield single[::width + 1]
            yield single[width - 1:len(single) - 1:width - 1]
        # z lines
        for i in range(width ** 2):
            yield tuple(single[i] for single in board)
        for j in range(width):
            # xz lines
            yield tuple(board[i][j * width + i] for i in range(len(board)))
            yield tuple(board[i][j * width + width - 1 - i]
                        for i in range(len(board)))
            # yz lines
            yield tuple(board[i][j + i * width] for i in range(len(board)))
            yield tuple(board[i][-j - 1 - i * width]
                        for i in range(len(board)))
        # xyz lines
        yield tuple(board[i][i * width + i] for i in range(len(board)))
        yield tuple(board[i][i * width + width - 1 - i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - width * (i + 1) + i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - (i * width) - i - 1]
                    for i in range(len(board)))

    @staticmethod
    def _to_string(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> str:
        """
        Return a string representation of the game board, in which integers
        represent the indices of empty spaces and the characters "X" and "O"
        represent previous move selections for MAX and MIN, repsectively.
        """
        display = "\n"
        for i in range(width):
            for j in range(width):
                line = board[j][i * width:i * width + width]
                start = j * width ** 2 + i * width
                for k, space in enumerate(line):
                    if space == 0:
                        space = start + k
                    else:
                        space = ("X" if space == 1
                                 else "O" if space == -1
                                 else "-")
                    display += "{0:>4}".format(space)
                display += " " * width
            display += "\n"
        return display


def get_children(parent: GameState) -> List[GameState]:
    children = list()
    for move in parent.moves:
        children.append(parent.traverse(move))
    return children


def get_ucb(x: Tuple[GameState, float]) -> float:
    return x[1]


def select_highest_ucb(parent: GameState,
                       board_dict: Dict, parent_child_dict: Dict) -> GameState:
    children: List[Tuple[GameState, float]] = list()
    if parent in parent_child_dict:
        children_list = parent_child_dict[parent]
    else:
        children_list = get_children(parent)
        parent_child_dict[parent] = children_list
    for child in children_list:
        if child not in board_dict:
            board_dict[child] = {'Parent': parent,
                                 'UCB': math.sqrt(2),
                                 'Wins': 0,
                                 'Attempts': 0}
        if board_dict[child]['Attempts'] > 0:
            board_dict[child]['UCB'] = \
                (board_dict[child]['Wins'] /
                    board_dict[child]['Attempts']) + \
                (math.sqrt(2) *
                    (math.sqrt(math.log10(
                        board_dict[board_dict[child]['Parent']]["Attempts"]) /
                        board_dict[child]['Attempts'])))
        children.append((child, board_dict[child]['UCB']))
    children.sort(key=get_ucb, reverse=True)
    return children[0][0]


def create_random_child(parent: GameState) -> GameState:
    moves = parent.moves
    rand_int = random.randrange(0, len(moves))
    return parent.traverse(moves[rand_int])


def simulate(node: GameState) -> int:
    curr = node
    while curr.util is None:
        moves = curr.moves
        rand_int = random.randrange(0, len(moves))
        curr = curr.traverse(moves[rand_int])
    return curr.util


def backpropagate(board_dict: Dict, sim_result: int,
                  last_child: GameState, player: int) -> None:
    curr = last_child
    while board_dict[curr]['Parent'] is not None:
        board_dict[curr]['Attempts'] += 1
        if sim_result == player:
            board_dict[curr]['Wins'] += 1
        elif sim_result == 0:
            board_dict[curr]['Wins'] += 0.5
        else:
            board_dict[curr]['Wins'] -= 1
        curr = board_dict[curr]['Parent']
    board_dict[curr]["Attempts"] += 1


def get_wr(x: Tuple[int, float]) -> float:
    return x[1]


def select_highest_wr(state: GameState, board_dict: Dict) -> int:
    next_moves: List[Tuple[int, float]] = list()
    for move in state.moves:
        next_state = state.traverse(move)
        next_moves.append((move, (board_dict[next_state]['Wins'] /
                                  board_dict[next_state]['Attempts'])
                                if board_dict[next_state]['Attempts']
                                     > 0 else 0))
    next_moves.sort(key=get_wr, reverse=True)
    return next_moves[0][0]


def find_best_move(state: GameState) -> None:
    """
    Search the game tree for the optimal move for the current player, storing
    the index of the move in the given GameState object's selected attribute.
    The move must be an integer indicating an index in the 3D board - ranging
    from 0 to 63 - with 0 as the index of the top-left space of the top board
    and 63 as the index of the bottom-right space of the bottom board.

    This function must perform a Monte Carlo Tree Search to select a move,
    using additional functions as necessary. During the search, whenever a
    better move is found, the selected attribute should be immediately updated
    for retrieval by the instructor's game driver. Each call to this function
    will be given a set number of seconds to run; when the time limit is
    reached, the index stored in selected will be used for the player's turn.
    """
    parent_child_dict: Dict = dict()
    board_dict: Dict = dict()
    board_dict[state] = {'Parent': None,
                         'UCB': math.sqrt(2),
                         'Wins': 0,
                         'Attempts': 0}

    while True:
        # Selection
        curr = state
        while curr.util is None and \
                (board_dict[curr]["Attempts"] > 0 or
                 board_dict[curr]["Parent"] is None):
            curr = select_highest_ucb(curr, board_dict, parent_child_dict)
        # Expansion
        if board_dict[curr]["Attempts"] == 0 and curr.util is None:
            random_child = create_random_child(curr)
            board_dict[random_child] = {'Parent': curr,
                                        'UCB': math.sqrt(2),
                                        'Wins': 0,
                                        'Attempts': 0}
            # Simulation
            sim_result = simulate(random_child)
            # Backpropagation
            backpropagate(board_dict, sim_result, random_child, state.player)
        else:
            # Simulation
            sim_result = curr.util
            # Backpropagation
            backpropagate(board_dict, sim_result, curr, state.player)
        state.selected = select_highest_wr(state, board_dict)


def main() -> None:
    board = ((0, 0, 0, 0,
              0, 0, None, None,
              0, None, 0, None,
              0, None, None, 0),) \
            + ((None,) * 16,) * 3
    board = ((0, 0, 0, 0,
              0, None, None, None,
              0, None, None, None,
              0, None, None, None),
             (None, None, None, None,
              None, None, None, None,
              None, None, None, None,
              None, None, None, None),
             (None, None, None, None,
              None, None, None, None,
              None, None, None, None,
              None, None, None, None),
             (0, 0, 0, 0,
              0, 0, None, None,
              0, None, 0, None,
              0, None, None, 0),)
    state = GameState(board, 1)
    print(state.display)
    find_best_move(state)
    assert state.selected == 0
#    play_game()


def play_game() -> None:
    """
    Play a game of 3D Tic-Tac-Toe with the computer.

    If you lose, you lost to a machine.
    If you win, your implementation was bad.
    You lose either way.
    """
    board = tuple(tuple(0 for _ in range(i, i + 16))
                  for i in range(0, 64, 16))
    state = GameState(board, 1)
    while state.util is None:
        # human move
        print(state.display)
        state = state.traverse(int(input("Move: ")))
        if state.util is not None:
            break
        # computer move
        find_best_move(state)
        move = (state.selected if state.selected != -1
                else random.choice(state.moves))
        state = state.traverse(move)
    print(state.display)
    if state.util == 0:
        print("Tie Game")
    else:
        print(f"Player {state.util} Wins!")


if __name__ == "__main__":
    main()
