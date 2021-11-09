# Name:         Kaanan Kharwa
# Name:         Parth Ray
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Know Op
# Term:         Summer 2021

import math
import itertools
import random
from typing import Callable, Dict, List, Tuple


class Math:
    """A collection of static methods for mathematical operations."""

    @staticmethod
    def dot(xs: List[float], ys: List[float]) -> float:
        """Return the dot product of the given vectors."""
        return sum(x * y for x, y in zip(xs, ys))

    @staticmethod
    def matmul(xs: List[List[float]],
               ys: List[List[float]]) -> List[List[float]]:
        """Multiply the given matrices and return the resulting matrix."""
        product = []
        for x_row in range(len(xs)):
            row = []
            for y_col in range(len(ys[0])):
                col = [ys[y_row][y_col] for y_row in range(len(ys))]
                row.append(Math.dot(xs[x_row], col))
            product.append(row)
        return product

    @staticmethod
    def transpose(matrix: List[List[float]]) -> List[List[float]]:
        """Return the transposition of the given matrix."""
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    @staticmethod
    def relu(z: float) -> float:
        """
        The activation function for hidden layers.
        """
        return z if z > 0 else 0.01 * z

    @staticmethod
    def relu_prime(z: float) -> float:
        """
        Return the derivative of the ReLU function.
        """
        return 1.0 if z > 0 else 0.0

    @staticmethod
    def sigmoid(z: float) -> float:
        """
        The activation function for the output layer.
        """
        epsilon = 1e-5
        return min(max(1 / (1 + math.e ** -z), epsilon), 1 - epsilon)

    @staticmethod
    def sigmoid_prime(z: float) -> float:
        """
        The activation function for the output layer.
        """
        return Math.sigmoid(z) * (1 - Math.sigmoid(z))

    @staticmethod
    def loss(actual: float, expect: float) -> float:
        """
        Return the loss between the actual and expected values.
        """
        return -(expect * math.log10(actual)
                 + (1 - expect) * math.log10(1 - actual))

    @staticmethod
    def loss_prime(actual: float, expect: float) -> float:
        """
        Return the derivative of the loss.
        """
        return -expect / actual + (1 - expect) / (1 - actual)


class Layer:  # do not modify class

    def __init__(self, size: Tuple[int, int], is_output: bool) -> None:
        """
        Create a network layer with size[0] levels and size[1] inputs at each
        level. If is_output is True, use the sigmoid activation function;
        otherwise, use the ReLU activation function.

        An instance of Layer has the following attributes.

            g: The activation function - sigmoid for the output layer and ReLU
               for the hidden layer(s).
            w: The weight matrix (randomly-initialized), where each inner list
               represents the incoming weights for one neuron in the layer.
            b: The bias vector (zero-initialized), where each value represents
               the bias for one neuron in the layer.
            z: The result of (wx + b) for each neuron in the layer.
            a: The activation g(z) for each neuron in the layer.
           dw: The derivative of the weights with respect to the loss.
           db: The derivative of the bias with respect to the loss.
        """
        self.g = Math.sigmoid if is_output else Math.relu
        self.w: List[List[float]] = \
            [[random.random() * 0.1 for _ in range(size[1])]
             for _ in range(size[0])]
        self.b: List[float] = [0.0] * size[0]

        # use of below attributes is optional but recommended
        self.z: List[float] = [0.0] * size[0]
        self.a: List[float] = [0.0] * size[0]
        self.dw: List[List[float]] = \
            [[0.0 for _ in range(size[1])] for _ in range(size[0])]
        self.db: List[float] = [0.0] * size[0]

    def __repr__(self) -> str:
        """
        Return a string representation of a network layer, with each level of
        the layer on a separate line, formatted as "W | B".
        """
        s = "\n"
        fmt = "{:7.3f}"
        for i in range(len(self.w)):
            s += " ".join(fmt.format(w) for w in self.w[i])
            s += " | " + fmt.format(self.b[i]) + "\n"
        return s

    def activate(self, inputs: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Given an input (x) of the same length as the number of columns in this
        layer's weight matrix, return g(wx + b).
        """
        self.z = [Math.dot(self.w[i], inputs) + self.b[i]
                   for i in range(len(self.w))]
        self.a = [self.g(real) for real in self.z]
        return tuple(self.a)


def create_samples(f: Callable[..., int], n_args: int, n_bits: int,
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Return a dictionary that maps inputs to expected outputs.
    """
    samples = {}
    max_arg = 2 ** n_bits
    for inputs in itertools.product((0, 1), repeat=n_args * n_bits):
        ints = [int("".join(str(bit) for bit in inputs[i:i + n_bits]), 2)
                for i in range(0, len(inputs), n_bits)]
        try:
            output = f(*ints)
            if 0 <= output < max_arg:
                bit_string = ("{:0" + str(n_bits) + "b}").format(output)
                samples[inputs] = tuple(int(bit) for bit in bit_string)
        except ZeroDivisionError:
            pass
    return samples


def forward_prop(layers: List[Layer], inputs: Tuple[float, ...])\
        -> Tuple[float, ...]:
    current_inputs = inputs
    for layer in layers:  # Assumes the input layer is not in the list
        current_inputs = layer.activate(current_inputs)
    return current_inputs  # This is the outputs


def g_prime(layer: Layer) -> List[float]:
    # For mult layers
    #if layer.g is Math.sigmoid:
    return [Math.sigmoid_prime(real) for real in layer.z]
    #return [Math.relu_prime(real) for real in layer.z]


def hadamard(list1: List[float], list2: List[float]) -> List[float]:
    new = []
    for num1, num2 in zip(list1, list2):
        new.append(num1 * num2)
    return new


def update_db(layer_db: List[float], new_db: List[float]) -> None:
    for i in range(len(layer_db)):
        layer_db[i] += new_db[i]


def update_dw(layer_dw: List[List[float]], new_dw: List[List[float]]) -> None:
    for i in range(len(layer_dw)):
        for j in range(len(layer_dw[i])):
            layer_dw[i][j] += new_dw[i][j]


def back_prop(da_init: List[List[float]], layers: List[Layer],
              inputs: List[float]) -> None:
    current_da = da_init
    for i in range(len(layers) - 1, -1, -1):
        layer = layers[i]
        new_db = hadamard(current_da[0], g_prime(layer))
        db = Math.transpose([new_db])
        new_dw = Math.matmul(db, [inputs] if i == 0 else [layers[i - 1].a])
    # For mult layers
    #    da_prev = Math.matmul(Math.transpose(layer.w), db)
    #    current_da = Math.transpose(da_prev)
        update_dw(layer.dw, new_dw)
        update_db(layer.db, new_db)


def update(layers: List[Layer], learning_rate: float, batch_size: int) -> None:
    for layer in layers:
        # For w
        for i in range(len(layer.w)):
            for j in range(len(layer.w[i])):
                layer.w[i][j] -= learning_rate * (layer.dw[i][j] / batch_size)
        # For b
        for i in range(len(layer.b)):
            layer.b[i] -= learning_rate * (layer.db[i] / batch_size)


def find_loss(output: Tuple[float, ...], actual: Tuple[float, ...])\
        -> List[float]:
    new = []
    for x in range(len(output)):
        new.append(Math.loss(output[x], actual[x]))
    return new


def find_loss_prime(output: Tuple[float, ...],
                    actual: Tuple[float, ...]) -> List[float]:
    new = []
    for x in range(len(output)):
        new.append(Math.loss_prime(output[x], actual[x]))
    return new


def reset(layers: List[Layer]) -> None:
    for layer in layers:
        layer.dw = [[0.0 for _ in range(len(layer.dw[0]))]
                    for _ in range(len(layer.dw))]
        layer.db = [0.0] * len(layer.db)


def train_network(samples: Dict[Tuple[int, ...], Tuple[int, ...]],
                  i_size: int, o_size: int) -> List[Layer]:
    """
    Given a training set (with labels) and the sizes of the input and output
    layers, create and train a network by iteratively propagating inputs
    (forward) and their losses (backward) to update its weights and biases.
    Return the resulting trained network.
    """
    layers = list()
    layers.append(Layer((o_size, i_size), True))
    sample_keys = list(samples.keys())
    batch_size = 40
    if i_size == o_size:
        learning_rate = .11
    else:
        learning_rate = .21
    while learning_rate > .00005:
        mini_batch = random.choices(sample_keys, k=batch_size)
        for sample in mini_batch:
            result = forward_prop(layers, sample)
            loss_prime = find_loss_prime(result, samples[sample])
            back_prop([loss_prime], layers, list(sample))
        update(layers, learning_rate, batch_size)
        reset(layers)
        learning_rate *= .99
    return layers


def main() -> None:

    random.seed(0)
    f = lambda x, y: x & y  # operation to learn
    n_args = 2              # arity of operation
    n_bits = 8              # size of each operand

    samples = create_samples(f, n_args, n_bits)
    train_pct = 0.95
    train_set = {inputs: samples[inputs]
               for inputs in random.sample(list(samples),
                                           k=int(len(samples) * train_pct))}
    test_set = {inputs: samples[inputs]
               for inputs in samples if inputs not in train_set}
    print("Train Size:", len(train_set), "Test Size:", len(test_set))

    network = train_network(train_set, n_args * n_bits, n_bits)
    for inputs in test_set:
        output = tuple(round(n, 2) for n in forward_prop(network, inputs))
        bits = tuple(round(n) for n in output)
        print("OUTPUT:", output)
        print("BITACT:", bits)
        print("BITEXP:", samples[inputs], end="\n\n")


if __name__ == "__main__":
    main()
