# Name:         Parth Ray
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Mine Shafted
# Term:         Summer 2021

import itertools
from typing import Callable, Generator, List, Tuple, Set, Dict


class BoardManager:  # do not modify

    def __init__(self, board: List[List[int]]):
        """
        An instance of BoardManager has two attributes.

            size: A 2-tuple containing the number of rows and columns,
                  respectively, in the game board.
            move: A callable that takes an integer as its only argument to be
                  used as the index to explore on the board. If the value at
                  that index is a clue (non-mine), this clue is returned;
                  otherwise, an error is raised.

        This constructor should only be called once per game.

        >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
        >>> bm = BoardManager(board)
        >>> bm.size
        (4, 3)
        >>> bm.move(4)
        2
        >>> bm.move(5)
        Traceback (most recent call last):
        ...
        RuntimeError
        """
        self.size: Tuple[int, int] = (len(board), len(board[0]))
        it: Generator[int, int, None] = BoardManager._move(board, self.size[1])
        next(it)
        self.move: Callable[[int], int] = it.send

    def get_adjacent(self, index: int) -> List[int]:
        """
        Return a list of indices adjacent (including diagonally) to the given
        index. All adjacent indices are returned, regardless of whether or not
        they have been explored.

        >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
        >>> bm = BoardManager(board)
        >>> bm.get_adjacent(3)
        [0, 1, 4, 6, 7]
        """
        row, col = index // self.size[1], index % self.size[1]
        return [i * self.size[1] + j
                for i in range(max(row - 1, 0), min(row + 2, self.size[0]))
                for j in range(max(col - 1, 0), min(col + 2, self.size[1]))
                if index != i * self.size[1] + j]

    @staticmethod
    def _move(board: List[List[int]], width: int) -> Generator[int, int, None]:
        """
        A generator that may be sent integers (indices to explore on the board)
        and yields integers (clues for the explored indices).

        Do not call this method directly; instead, call the |move| instance
        attribute, which sends its index argument to this generator.
        """
        index = (yield 0)
        while True:
            clue = board[index // width][index % width]
            if clue == -1:
                raise RuntimeError
            index = (yield clue)


def get_unexplored_list(explored_list: Set, adj_list: List[int]) -> List[int]:
    return list(set(adj_list) - explored_list)


def get_domain(clue: int, unexplored_adj_list: List[int]) \
        -> List[Tuple[int, ...]]:
    list_len = len(unexplored_adj_list)
    domain: List[Tuple[int, ...]] = list()
    for i in itertools.permutations(range(list_len), list_len - clue):
        temp = unexplored_adj_list.copy()
        for j in i:
            temp[j] = -unexplored_adj_list[j]
        domain.append(tuple(temp))
    return list(set(domain))


def produce_arcs(idx1: int, adj_list1: List[Tuple[int, ...]], idx2: int,
                 adj_list2: List[Tuple[int, ...]]) -> Tuple[int, int]:
    for i in adj_list1[0]:
        for j in adj_list2[0]:
            if abs(i) == abs(j):
                return idx1, idx2
    return -1, -1


def arc_reduce(arc: Tuple[int, int], domain_dict: Dict) -> Tuple[bool, Dict]:
    change: bool = False
    shared_idx: List[int] = list()
    for i in domain_dict[arc[0]][0]:
        for j in domain_dict[arc[1]][0]:
            if abs(i) == abs(j):
                shared_idx.append(i)
    num_shared_idx = len(shared_idx)
    new_domain: List[Tuple[int]] = list()
    for domain_x in domain_dict[arc[0]]:
        valid: bool = False
        for domain_y in domain_dict[arc[1]]:
            if (len(domain_x) - num_shared_idx) == \
                    len(set(domain_x) - set(domain_y)):
                valid = True
        if not valid:
            change = True
        else:
            new_domain.append(domain_x)
    domain_dict[arc[0]] = new_domain
    return change, domain_dict


def get_explorable_idx(domain_dict: Dict, explored_set: Set, mine_set: Set)\
        -> Tuple[Set[int], Set[int]]:
    explorable_set: Set = set()
    for domain in domain_dict.values():
        if len(domain) == 1:
            for idx in domain[0]:
                if idx not in explored_set and idx not in explorable_set:
                    if idx < 0:
                        explorable_set.add(abs(idx))
                    else:
                        mine_set.add(idx)
        else:  # find consistent idx
            for idx in domain[0]:
                inall = True
                if idx not in explored_set and idx not in explorable_set:
                    for other in domain[1:]:
                        inall = inall and (idx in other)
                    if inall:
                        if idx < 0:
                            explorable_set.add(abs(idx))
                        else:
                            mine_set.add(idx)
    return mine_set, explorable_set


def create_solution_board(bm: BoardManager, explored_set: Set) \
        -> List[List[int]]:
    solution: List[List[int]] = [[-1] * bm.size[1] for i in range(bm.size[0])]
    for idx in explored_set:
        row, col = idx // bm.size[1], idx % bm.size[1]
        solution[row][col] = bm.move(idx)
    return solution


def sweep_mines(bm: BoardManager) -> List[List[int]]:
    """
    Given a BoardManager (bm) instance, return a solved board (represented as a
    2D list) by repeatedly calling bm.move(index) until all safe indices have
    been explored. If at any time a move is attempted on a non-safe index, the
    BoardManager will raise an error; this error signifies the end of the game
    and should not attempt to be caught.

    >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    >>> bm = BoardManager(board)
    >>> sweep_mines(bm)
    [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    """
    board_len = bm.size[0] * bm.size[1]
    domain_dict: Dict = dict()
    explored_set: Set = set()
    mine_set: Set = set()
    arcs_list: List[Tuple[int, int]] = list()
    frontier_idx: Set = set()
    frontier_idx.add(0)
    while len(explored_set) + len(mine_set) < board_len:
        for idx in frontier_idx:
            clue = bm.move(idx)
            explored_set.add(idx)
            unexplored_adj_list = get_unexplored_list(explored_set,
                                                      bm.get_adjacent(idx))
            domain = get_domain(clue, unexplored_adj_list)
            domain_dict[idx] = domain
        for i in itertools.combinations(explored_set, 2):
            arcs = produce_arcs(i[0], domain_dict[i[0]],
                                i[1], domain_dict[i[1]])
            if arcs != (-1, -1):
                arcs_list.append(arcs)
                arcs_list.append((arcs[1], arcs[0]))
        agenda = arcs_list.copy()
        while len(agenda) != 0:
            arc = agenda.pop()
            change, domain_dict = arc_reduce(arc, domain_dict)
            if change:
                for i in arcs_list:
                    if arc[0] == i[1]:
                        agenda.append(i)
        mine_set, frontier_idx = get_explorable_idx(domain_dict,
                                                    explored_set, mine_set)
    return create_solution_board(bm, explored_set)


def main() -> None:  # optional driver
    board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    bm = BoardManager(board)
    assert sweep_mines(bm) == board


if __name__ == "__main__":
    main()
