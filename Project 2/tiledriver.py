# Name:         Parth Ray
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver I
# Term:         Summer 2021

import queue
import math
import random
from typing import List, Tuple


class Heuristic:

    @staticmethod
    def get(tiles: Tuple[int, ...]) -> int:
        """
        Return the estimated distance to the goal using Manhattan distance
        and linear conflicts.

        Only this static method should be called during a search; all other
        methods in this class should be considered private.

        >>> Heuristic.get((0, 1, 2, 3))
        0
        >>> Heuristic.get((3, 2, 1, 0))
        6
        """
        width = int(len(tiles) ** 0.5)
        return (Heuristic._get_manhattan_distance(tiles, width)
                + Heuristic._get_linear_conflicts(tiles, width))

    @staticmethod
    def _get_manhattan_distance(tiles: Tuple[int, ...], width: int) -> int:
        """
        Return the Manhattan distance of the given tiles, which represents
        how many moves is tile is away from its goal position.
        """
        distance = 0
        for i in range(len(tiles)):
            if tiles[i] != 0:
                row_dist = abs(i // width - tiles[i] // width)
                col_dist = abs(i % width - tiles[i] % width)
                distance += row_dist + col_dist
        return distance

    @staticmethod
    def _get_linear_conflicts(tiles: Tuple[int, ...], width: int) -> int:
        """
        Return the number of linear conflicts in the tiles, which represents
        the minimum number of tiles in each row and column that must leave and
        re-enter that row or column in order for the puzzle to be solved.
        """
        conflicts = 0
        rows = [[] for i in range(width)]
        cols = [[] for i in range(width)]
        for i in range(len(tiles)):
            if tiles[i] != 0:
                if i // width == tiles[i] // width:
                    rows[i // width].append(tiles[i])
                if i % width == tiles[i] % width:
                    cols[i % width].append(tiles[i])
        for i in range(width):
            conflicts += Heuristic._count_conflicts(rows[i])
            conflicts += Heuristic._count_conflicts(cols[i])
        return conflicts * 2

    @staticmethod
    def _count_conflicts(ints: List[int]) -> int:
        """
        Return the minimum number of tiles that must be removed from the given
        list in order for the list to be sorted.
        """
        if Heuristic._is_sorted(ints):
            return 0
        lowest = None
        for i in range(len(ints)):
            conflicts = Heuristic._count_conflicts(ints[:i] + ints[i + 1:])
            if lowest is None or conflicts < lowest:
                lowest = conflicts
        return 1 + lowest

    @staticmethod
    def _is_sorted(ints: List[int]) -> bool:
        """Return True if the given list is sorted and False otherwise."""
        for i in range(len(ints) - 1):
            if ints[i] > ints[i + 1]:
                return False
        return True


class State:

    def __init__(self, path: str, tiles: Tuple[int, ...]) -> None:
        self.path = path
        self.tiles = tiles
        self.heuristic = Heuristic.get(tiles)
        self.lc = Heuristic._get_linear_conflicts\
            (tiles, int(len(tiles) ** 0.5))

    def __eq__(self, other) -> bool:
        return self.tiles == other.tiles

    def __repr__(self) -> str:
        return "State path: %s    tiles: %s     heuristic: %d" \
               % (self.path, self.tiles, self.heuristic)

    def __hash__(self) -> int:
        return hash(self.tiles)

    def __lt__(self, other) -> bool:
        return (self.heuristic + len(self.path)) < \
               (other.heuristic + len(self.path))

    def random_tiles(self):
        temp = list(self.tiles)
        random.shuffle(temp)
        self.tiles = tuple(temp)
        self.heuristic = Heuristic.get(self.tiles)
        self.lc = Heuristic._get_linear_conflicts\
            (self.tiles, int(len(self.tiles) ** 0.5))
        self.path = ""


def create_frontier_states(state: State) -> List[State]:
    width = int(math.sqrt(len(state.tiles)))
    blank = state.tiles.index(0)
    row = blank // width
    col = blank % width
    new_states = []
    """move left:Cannot have previously gone right
                 Blank cannot be in rightmost column"""
    if (state.path == "" or state.path[-1] != "L") and col != width - 1:
        new_tiles = list(state.tiles)
        new_tiles[blank], new_tiles[blank + 1] = \
            new_tiles[blank + 1], new_tiles[blank]
        new_states.append(State(state.path + "H", tuple(new_tiles)))

    """move right:Cannot have previously gone left
                  Blank cannot be in leftmost column"""
    if (state.path == "" or state.path[-1] != "H") and col != 0:
        new_tiles = list(state.tiles)
        new_tiles[blank], new_tiles[blank - 1] = \
            new_tiles[blank - 1], new_tiles[blank]
        new_states.append(State(state.path + "L", tuple(new_tiles)))

    """move up:Cannot have previously gone down
               Blank cannot be in last row"""
    if (state.path == "" or state.path[-1] != "J") and row != width - 1:
        new_tiles = list(state.tiles)
        new_tiles[blank], new_tiles[blank + width] = \
            new_tiles[blank + width], new_tiles[blank]
        new_states.append(State(state.path + "K", tuple(new_tiles)))

    """move down:Cannot have previously gone up
                 Blank cannot be in first row"""
    if (state.path == "" or state.path[-1] != "K") and row != 0:
        new_tiles = list(state.tiles)
        new_tiles[blank], new_tiles[blank - width] = \
            new_tiles[blank - width], new_tiles[blank]
        new_states.append(State(state.path + "J", tuple(new_tiles)))
    
    return new_states


def solve_puzzle(tiles: Tuple[int, ...]) -> str:
    """
    Return a string (containing characters "H", "J", "K", "L") representing the
    optimal number of moves to solve the given puzzle.
    """
    initial_state = State("", tiles)
    if initial_state.heuristic == 0:
        return initial_state.path
    que: queue.PriorityQueue = queue.PriorityQueue()
    que.put((initial_state.heuristic + len(initial_state.path), initial_state))
    while not que.empty():
        parent = que.get()[1]
        if parent.heuristic == 0:
            return parent.path
        children = create_frontier_states(parent)
        for child in children:
            que.put((child.heuristic + len(child.path), child))
    return "No Solution Found"


def main() -> None:
    pass  # optional program test driver


if __name__ == "__main__":
    main()
