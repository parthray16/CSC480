# Name:         Parth Ray
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Moonlander II
# Term:         Summer 2021

import random
from typing import Callable, Tuple, Dict


class ModuleState:  # do not modify class

    def __init__(self, fuel: int, altitude: float, force: float,
                 transition: Callable[[float, float], float],
                 velocity: float = 0.0,
                 actions: Tuple[int, ...] = tuple(range(5))) -> None:
        """
        An instance of ModuleState has the following attributes.

            fuel: The amount of fuel (in liters) able to be used.
            altitude: The distance (in meters) of the module from the surface
                      of its target object.
            velocity: The speed of the module, where a positive value indicates
                      movement away from the target object and a negative value
                      indicates movement toward it. Defaults to zero.
            actions: The available fuel rates, where 0 indicates free-fall and
                     the highest-valued action indicates maximum thrust away
                     from the target object. Defaults to (0, 1, 2, 3, 4).
            use_fuel: A callable that takes an integer as its only argument to
                      be used as the fuel rate for moving to the next state.
        """
        self.fuel: int = fuel
        self.altitude: float = altitude
        self.velocity: float = velocity
        self.actions: Tuple[int, ...] = actions
        self.use_fuel: Callable[[int], ModuleState] = \
            lambda rate: self._use_fuel(force, transition, rate)

    def __repr__(self) -> str:
        if not self.altitude:
            return ("-" * 16 + "\n"
                    + f" Remaining Fuel: {self.fuel:4} l\n"
                    + f"Impact Velocity: {self.velocity:7.2f} m/s\n")
        else:
            return (f"    Fuel: {self.fuel:4} l\n"
                    + f"Altitude: {self.altitude:7.2f} m\n"
                    + f"Velocity: {self.velocity:7.2f} m/s\n")

    def set_actions(self, n: int) -> None:
        """
        Set the number of actions available to the module simulator, which must
        be at least two. Calling this method overrides the default number of
        actions set in the constructor.

        >>> module.set_actions(8)
        >>> module.actions
        (0, 1, 2, 3, 4, 5, 6, 7)
        """
        if n < 2:
            raise ValueError
        self.actions = tuple(range(n))

    def _use_fuel(self, force: float, transition: Callable[[float, int], float],
                  rate: int) -> "ModuleState":
        """
        Return a ModuleState instance in which the fuel, altitude, and velocity
        are updated based on the fuel rate chosen.

        Do not call this method directly; instead, call the |use_fuel| instance
        attribute, which only requires a fuel rate as its argument.
        """
        if not self.altitude:
            return self
        fuel = max(0, self.fuel - rate)
        if not fuel:
            rate = 0
        acceleration = transition(force * 9.8, rate / (len(self.actions) - 1))
        altitude = max(0.0, self.altitude + self.velocity + acceleration / 2)
        velocity = self.velocity + acceleration
        return ModuleState(fuel, altitude, force, transition, velocity=velocity,
                           actions=self.actions)


def get_reward(state: ModuleState) -> float:
    if state.altitude == 0:
        if -1 < state.velocity < 0:
            return 1
        return -1
    if state.velocity >= 0:
        return -.04
    return -0.04


def choose_action(state: ModuleState, q_table: Dict, eps: float,
                  max_alt: float) -> int:
    if random.uniform(0, 1) < eps:
        return random.choice(state.actions)
    return q_table[simplify_state(state, max_alt)].index\
        (max(q_table[simplify_state(state, max_alt)]))


def simplify_state(state: ModuleState, max_alt: float) -> Tuple[float, float]:
    if state.altitude > max_alt:
        altitude = max_alt
    elif state.altitude > 0:
        altitude = round(state.altitude, 2)
        if altitude == 0:
            altitude = 1
    else:
        altitude = 0
    return altitude, state.velocity


def calc_bellman(learning_rate: float, discount_rate: float, reward: float,
                 current_q: float, next_q: float) -> float:
    return (1 - learning_rate) * current_q + \
           learning_rate * (reward + discount_rate * next_q)


def check_n_add(q_table: Dict, state: ModuleState,
                max_alt: float) -> None:
    simple_state = simplify_state(state, max_alt)
    if state.altitude == 0 and simple_state not in q_table:
        q_table[simple_state] = [get_reward(state)] * len(state.actions)
    if simple_state not in q_table:
        q_table[simple_state] = [0] * len(state.actions)


def learn_q(state: ModuleState) -> Callable[[ModuleState, int], float]:
    """
    Return a Q-function that maps a state-action pair to a utility value. This
    function must be a callable with the signature (ModuleState, int) -> float.

    Optional: Use |state.set_actions| to set the size of the action set. Higher
    values offer more control (sensitivity to differences in rate changes), but
    require larger Q-tables and thus more training time.
    """
    q_table: Dict = dict()
    max_alt = state.altitude
    q = lambda s, a: q_table[simplify_state(s, max_alt)][a]
    esp = 1
    lr = 1
    episodes = 20000
    for _ in range(episodes):
        curr_state = state
        if esp > .1:
            esp -= .001
        if lr > .30:
            lr -= .001
        while curr_state.altitude != 0:
            simple_state = simplify_state(curr_state, max_alt)
            check_n_add(q_table, curr_state, max_alt)
            # esp greedily choose an action
            action = choose_action(curr_state, q_table, esp, max_alt)
            # get curr_state reward
            reward = get_reward(curr_state)
            # create next state with action
            next_state = curr_state.use_fuel(action)
            simple_next_state = simplify_state(next_state, max_alt)
            check_n_add(q_table, next_state, max_alt)
            # update curr_state, action util
            q_table[simple_state][action] = calc_bellman(lr, 1, reward,
                q_table[simple_state][action], max(q_table[simple_next_state]))
            # go next
            curr_state = next_state
    return q


def main() -> None:
    fuel: int = 1000
    altitude: float = 100.0

    gforces = {"Pluto": 0.063, "Moon": 0.1657, "Mars": 0.378, "Venus": 0.905,
               "Earth": 1.0, "Jupiter": 2.528}
    transition = lambda g, r: g * (2 * r - 1)  # example transition function
    state = ModuleState(fuel, altitude, gforces["Earth"], transition)
    init = ModuleState(fuel, altitude, gforces["Earth"], transition)
    while True:
        state = ModuleState(fuel, altitude, gforces["Earth"], transition)
        q = learn_q(state)
        policy = lambda s: max(state.actions, key=lambda a: q(s, a))
        while state.altitude > 0:
            state = state.use_fuel(policy(state))
        if state.velocity < -1:
            break
    state = init
    print(state)
    while state.altitude > 0:
        print(policy(state))
        state = state.use_fuel(policy(state))
        print(state)


if __name__ == "__main__":
    main()
