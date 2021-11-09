# Name:         Parth Ray
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver II
# Term:         Summer 2021
import queue
import random
from typing import Callable, List, Tuple

import tiledriver


def is_solvable(tiles: Tuple[int, ...]) -> bool:
    """
    Return True if the given tiles represent a solvable puzzle and False
    otherwise.

    >>> is_solvable((3, 2, 1, 0))
    True
    >>> is_solvable((0, 2, 1, 3))
    False
    """
    _, inversions = _count_inversions(list(tiles))
    width = int(len(tiles) ** 0.5)
    if width % 2 == 0:
        row = tiles.index(0) // width
        return (row % 2 == 0 and inversions % 2 == 0 or
                row % 2 == 1 and inversions % 2 == 1)
    else:
        return inversions % 2 == 0


def _count_inversions(ints: List[int]) -> Tuple[List[int], int]:
    """
    Count the number of inversions in the given sequence of integers (ignoring
    zero), and return the sorted sequence along with the inversion count.

    This function is only intended to assist |is_solvable|.

    >>> _count_inversions([3, 7, 1, 4, 0, 2, 6, 8, 5])
    ([1, 2, 3, 4, 5, 6, 7, 8], 10)
    """
    if len(ints) <= 1:
        return ([], 0) if 0 in ints else (ints, 0)
    midpoint = len(ints) // 2
    l_side, l_inv = _count_inversions(ints[:midpoint])
    r_side, r_inv = _count_inversions(ints[midpoint:])
    inversions = l_inv + r_inv
    i = j = 0
    sorted_tiles = []
    while i < len(l_side) and j < len(r_side):
        if l_side[i] <= r_side[j]:
            sorted_tiles.append(l_side[i])
            i += 1
        else:
            sorted_tiles.append(r_side[j])
            inversions += len(l_side[i:])
            j += 1
    sorted_tiles += l_side[i:] + r_side[j:]
    return (sorted_tiles, inversions)


def create_base_puzzle(width: int) -> List[int]:
    start_tuple = list()
    for j in range(width):
        for i in range(0, width):
            start_tuple.append(i + j + ((width - 1) * j))
    return start_tuple


def get_best_neighbor_lc(neighbors: List[tiledriver.State])\
        -> tiledriver.State:
    best_neighbor = neighbors[0]
    for i in range(1, len(neighbors)):
        if neighbors[i].lc > best_neighbor.lc:
            best_neighbor = neighbors[i]
    return best_neighbor


def explore_plateau(plat_state: tiledriver.State,
                    neighbors: List[tiledriver.State],
                    count: int) -> tiledriver.State:
    que: queue.Queue = queue.Queue()
    for neighbor in neighbors:
        que.put((neighbor.lc, neighbor))
    while not que.empty() and count > 0:
        state = que.get()
        if state[0] > plat_state.lc:
            return state[1]
        best_neighbor = get_best_neighbor_lc\
            (tiledriver.create_frontier_states(state[1]))
        que.put((best_neighbor.lc, best_neighbor))
        count -= 1
    while not que.empty():
        state = que.get()
        if state[0] > plat_state.lc:
            return state[1]
    return plat_state


def conflict_tiles(width: int, min_lc: int) -> Tuple[int, ...]:
    """
    Create a solvable shuffled puzzle of the given width with a minimum number
    of linear conflicts (ignoring Manhattan distance).

    >>> tiles = conflict_tiles(3, 5)
    >>> tiledriver.Heuristic._get_linear_conflicts(tiles, 3)
    5
    """
    start_tuple = create_base_puzzle(width)
    random.shuffle(start_tuple)
    curr_state = tiledriver.State("", tuple(start_tuple))
    while not is_solvable(curr_state.tiles):
        curr_state.random_tiles()
    while curr_state.lc < min_lc:
        neighbors = tiledriver.create_frontier_states(curr_state)
        best_neighbor = get_best_neighbor_lc(neighbors)
        if curr_state.lc < best_neighbor.lc:
            curr_state = best_neighbor
            continue
        if curr_state.lc == best_neighbor.lc:
            curr_state = explore_plateau(curr_state, neighbors, 10)
        if curr_state.lc >= best_neighbor.lc:
            curr_state.random_tiles()
            while not is_solvable(curr_state.tiles):
                curr_state.random_tiles()
    return curr_state.tiles


def shuffle_tiles(width: int, min_len: int,
                  solve_puzzle: Callable[[Tuple[int, ...]], str]
                  ) -> Tuple[int, ...]:
    """
    Create a solvable shuffled puzzle of the given width with an optimal
    solution length equal to or greater than the given minimum length.

    >>> tiles = shuffle_tiles(3, 6, tiledriver.solve_puzzle)
    >>> len(tiledriver.solve_puzzle(tiles))
    6
    """
    start_tuple = create_base_puzzle(width)
    random.shuffle(start_tuple)
    curr_state = tiledriver.State("", tuple(start_tuple))
    while not is_solvable(curr_state.tiles):
        curr_state.random_tiles()
    curr_state.path = solve_puzzle(curr_state.tiles)
    while len(curr_state.path) < min_len:
        next_moves = tiledriver.create_frontier_states(curr_state)
        parent = curr_state
        for child in next_moves:
            if len(curr_state.path) > child.heuristic:
                continue
            child.path = solve_puzzle(child.tiles)
            if len(curr_state.path) < len(child.path):
                curr_state = child
        if parent == curr_state:
            curr_state.random_tiles()
            while True:
                if curr_state.heuristic >= (.8 * min_len) and \
                        is_solvable(curr_state.tiles):
                    break
                curr_state.random_tiles()
        curr_state.path = solve_puzzle(curr_state.tiles)
    return curr_state.tiles


def main() -> None:
    """tiles = conflict_tiles(3, 10)
    print(tiledriver.Heuristic._get_linear_conflicts(tiles, 3))
    tiles = conflict_tiles(4, 14)
    print(tiledriver.Heuristic._get_linear_conflicts(tiles, 4))
    tiles = conflict_tiles(5, 18)
    print(tiledriver.Heuristic._get_linear_conflicts(tiles, 5))
    tiles = shuffle_tiles(2, 6, tiledriver.solve_puzzle)
    print(len(tiledriver.solve_puzzle(tiles)))
    tiles = shuffle_tiles(3, 29, tiledriver.solve_puzzle)
    print(len(tiledriver.solve_puzzle(tiles)))"""


if __name__ == "__main__":
    main()
