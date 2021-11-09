# Name:         Parth Ray
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Biogimmickry
# Term:         Summer 2021

import random
from typing import Callable, Dict, Tuple, List


class FitnessEvaluator:

    def __init__(self, array: Tuple[int, ...]) -> None:
        """
        An instance of FitnessEvaluator has one attribute, which is a callable.

            evaluate: A callable that takes a program string as its only
                      argument and returns an integer indicating how closely
                      the program populated the target array, with a return
                      value of zero meaning the program was accurate.

        This constructor should only be called once per search.

        >>> fe = FitnessEvaluator((0, 20))
        >>> fe.evaulate(">+")
        19
        >>> fe.evaulate("+++++[>++++<-]")
        0
        """
        self.evaluate: Callable[[str], int] = \
            lambda program: FitnessEvaluator._evaluate(array, program)

    @staticmethod
    def interpret(program: str, size: int) -> Tuple[int, ...]:
        """
        Using a zeroed-out memory array of the given size, run the given
        program to update the integers in the array. If the program is
        ill-formatted or requires too many iterations to interpret, raise a
        RuntimeError.
        """
        p_ptr = 0
        a_ptr = 0
        count = 0
        max_steps = 1000
        i_map = FitnessEvaluator._preprocess(program)
        memory = [0] * size
        while p_ptr < len(program):
            if program[p_ptr] == "[":
                if memory[a_ptr] == 0:
                    p_ptr = i_map[p_ptr]
            elif program[p_ptr] == "]":
                if memory[a_ptr] != 0:
                    p_ptr = i_map[p_ptr]
            elif program[p_ptr] == "<":
                if a_ptr > 0:
                    a_ptr -= 1
            elif program[p_ptr] == ">":
                if a_ptr < len(memory) - 1:
                    a_ptr += 1
            elif program[p_ptr] == "+":
                memory[a_ptr] += 1
            elif program[p_ptr] == "-":
                memory[a_ptr] -= 1
            else:
                raise RuntimeError
            p_ptr += 1
            count += 1
            if count > max_steps:
                raise RuntimeError
        return tuple(memory)

    @staticmethod
    def _preprocess(program: str) -> Dict[int, int]:
        """
        Return a dictionary mapping the index of each [ command with its
        corresponding ] command. If the program is ill-formatted, raise a
        RuntimeError.
        """
        i_map = {}
        stack = []
        for p_ptr in range(len(program)):
            if program[p_ptr] == "[":
                stack.append(p_ptr)
            elif program[p_ptr] == "]":
                if len(stack) == 0:
                    raise RuntimeError
                i = stack.pop()
                i_map[i] = p_ptr
                i_map[p_ptr] = i
        if len(stack) != 0:
            raise RuntimeError
        return i_map

    @staticmethod
    def _evaluate(expect: Tuple[int, ...], program: str) -> int:
        """
        Return the sum of absolute differences between each index in the given
        tuple and the memory array created by interpreting the given program.
        """
        actual = FitnessEvaluator.interpret(program, len(expect))
        return sum(abs(x - y) for x, y in zip(expect, actual))


def create_random_program(max_len: int) -> str:
    alphabet = ['<', '>', '+', '-']
    alphabet_len = len(alphabet)
    if max_len == 0:
        min_len = 1
    else:
        min_len = 12
    random_len = random.randint(
                    min_len,
                    max_len if max_len > 0 else random.randint(8, 24))
    # create space for loop braces
    if max_len > 0:
        random_len -= 2
    program = ''
    for _ in range(random_len):
        program += alphabet[random.randint(0, alphabet_len - 1)]
    if max_len > 0:
        return insert_loop(program)
    return program


def insert_loop(program: str) -> str:
    prog_len = len(program)
    open_brace_pos = random.randint(1, prog_len - 6)
    close_brace_pos = random.randint(open_brace_pos + 5, prog_len - 1)
    return program[:open_brace_pos] + '[' \
            + program[open_brace_pos:close_brace_pos] + ']'\
            + program[close_brace_pos:]


def crossover_no_loop(indiv1: str, indiv2: str) -> Tuple[str, str]:
    smaller_len = min(len(indiv1), len(indiv2))
    crossover_idx = random.randint(1, smaller_len)
    return (indiv1[:crossover_idx] + indiv2[crossover_idx:],
            indiv2[:crossover_idx] + indiv1[crossover_idx:])


def crossover_loop(indiv1: str, indiv2: str) -> Tuple[str, str]:
    smaller_len = min(len(indiv1), len(indiv2))
    open_brace_pos1 = indiv1.index('[')
    close_brace_pos1 = indiv1.index(']')
    open_brace_pos2 = indiv2.index('[')
    close_brace_pos2 = indiv2.index(']')
    crossover_idx = random.randint(0, smaller_len)
    while True:
        if crossover_idx < open_brace_pos1 and crossover_idx < open_brace_pos2:
            break
        if crossover_idx > close_brace_pos1 and \
                crossover_idx > close_brace_pos2:
            break
        if open_brace_pos1 < crossover_idx < close_brace_pos1 and \
                open_brace_pos2 < crossover_idx < close_brace_pos2:
            break
        crossover_idx = random.randint(0, smaller_len)
    return (indiv1[:crossover_idx] + indiv2[crossover_idx:],
            indiv2[:crossover_idx] + indiv1[crossover_idx:])


def mutate(indiv1: str, indiv2: str, max_len: int) -> Tuple[str, str]:
    if random.randint(1, 100) < 35:
        if max_len == 0:
            indiv1 = mutate_no_loop(indiv1)
            indiv2 = mutate_no_loop(indiv2)
        else:
            indiv1 = mutate_loop(indiv1, max_len)
            indiv2 = mutate_loop(indiv2, max_len)
    return indiv1, indiv2


def mutate_no_loop(indiv: str) -> str:
    alphabet = ['<', '>', '+', '-']
    alphabet_len = len(alphabet)
    indiv_len = len(indiv)
    num_mutations = 1 + indiv_len // 8
    for _ in range(num_mutations):
        rand_int = random.randint(1, 3)
        mutation_idx = random.randint(0, indiv_len - 1)
        alphabet_idx = random.randint(0, alphabet_len - 1)
        # addition
        if rand_int == 1:
            indiv = indiv[:mutation_idx] + \
                    alphabet[alphabet_idx] + indiv[mutation_idx:]
        # deletion
        if rand_int == 2:
            if indiv_len > 1:
                indiv = indiv[:mutation_idx] + indiv[mutation_idx + 1:]
        # edit
        if rand_int == 3:
            indiv = indiv[:mutation_idx] + \
                    alphabet[alphabet_idx] + indiv[mutation_idx + 1:]
    return indiv


def mutate_loop(indiv: str, max_len: int) -> str:
    alphabet = ['<', '>', '+', '-']
    alphabet_len = len(alphabet)
    indiv_len = len(indiv)
    num_mutations = 1 + indiv_len // 8
    for _ in range(num_mutations):
        rand_int = random.randint(1, 3)
        mutation_idx = random.randint(0, indiv_len - 1)
        alphabet_idx = random.randint(0, alphabet_len - 1)
        # addition
        if rand_int == 1:
            if indiv_len + 1 <= max_len:
                indiv = indiv[:mutation_idx] + \
                    alphabet[alphabet_idx] + indiv[mutation_idx:]
                indiv_len += 1
            else:
                rand_int = 2
        # deletion
        if rand_int == 2:
            if mutation_idx == 0 and indiv[mutation_idx + 1] == '[':
                rand_int = 3
            elif indiv_len - 1 >= 12 and \
                    indiv[mutation_idx] != '[' and indiv[mutation_idx] != ']':
                indiv = indiv[:mutation_idx] + indiv[mutation_idx + 1:]
                indiv_len -= 1
        # edit
        if rand_int == 3:
            if indiv[mutation_idx] != '[' and indiv[mutation_idx] != ']':
                indiv = indiv[:mutation_idx] + \
                    alphabet[alphabet_idx] + indiv[mutation_idx + 1:]
    return indiv


def check_converges(top_20: List[Tuple[str, int]]) -> bool:
    if len(top_20) == 2:
        return top_20[0][0] == top_20[1][0]
    return top_20[0][0] == top_20[1][0] and check_converges(top_20[2:])


def initialize_population(fe: FitnessEvaluator, pop_size: int, max_len: int)\
        -> List[Tuple[str, int]]:
    generation: List[Tuple[str, int]] = list()
    for _ in range(pop_size):
        individual = create_random_program(max_len)
        fitness = -1
        while fitness < 0:
            try:
                fitness = fe.evaluate(individual)
            except RuntimeError:
                individual = create_random_program(max_len)
                fitness = -1
        generation.append((individual, fitness))
    return generation


def create_program(fe: FitnessEvaluator, max_len: int) -> str:
    """
    Return a program string no longer than max_len that, when interpreted,
    populates a memory array that exactly matches a target array.

    Use fe.evaluate(program) to get a program's fitness score (zero is best).
    """
    converged = True
    pop_size = 1200
    percentile = .10
    weights = [i for i in range(int(pop_size * percentile), 0, -1)]
    while True:
        if converged:
            converged = False
            generation = initialize_population(fe, pop_size, max_len)
            generation.sort(key=lambda indiv: indiv[1])
        if generation[0][1] == 0:
            return generation[0][0]
        next_gen: List[Tuple[str, int]] = list()
        while len(next_gen) < pop_size:
            indiv1_, indiv2_ = random.choices(
                generation[:int(pop_size * percentile)]
                , weights=weights, k=2)
            if max_len == 0:
                indiv1, indiv2 = crossover_no_loop(indiv1_[0], indiv2_[0])
            else:
                indiv1, indiv2 = crossover_loop(indiv1_[0], indiv2_[0])
            indiv1, indiv2 = mutate(indiv1, indiv2, max_len)
            try:
                fitness1 = fe.evaluate(indiv1)
                fitness2 = fe.evaluate(indiv2)
            except RuntimeError:
                continue

            next_gen.append((indiv1, fitness1))
            next_gen.append((indiv2, fitness2))
        next_gen.sort(key=lambda indiv: indiv[1])
        if check_converges(next_gen[0:20]):
            converged = True
        else:
            generation = next_gen


def main() -> None:  # optional driver
    array = (1, 2, -3, 4, 1, 2, -3, 4)
    max_len = 0  # no BF loop required
    # only attempt when non-loop programs work
    array = (20, 0)
    max_len = 15
    program = create_program(FitnessEvaluator(array), max_len)
    if max_len > 0:
        assert len(program) <= max_len
    assert array == FitnessEvaluator.interpret(program, len(array))
    print(program)
    print(FitnessEvaluator.interpret(program, len(array)))


if __name__ == "__main__":
    main()
