import numpy as np
from collections import defaultdict
from typing import Callable, Tuple
import random



def create_greedy_policy(Q: defaultdict) -> Callable:
    """Creates an initial blackjack policy from default_blackjack_policy but updates policy using Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """

    def get_action(state: Tuple) -> int:
        # If state was never seen before, use initial blackjack policy
        if state not in Q.keys():
            return default_blackjack_policy(state)
        else:
            # Choose deterministic greedy action
            chosen_action = np.argmax(Q[state]).item()
            return chosen_action

    return get_action


def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])
    # print("num_actions = ",num_actions)
    def get_action(state: Tuple) -> int:
        # TODO
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        # print("Num actions = ",num_actions)
        # print("Q[state] = ",Q[state])
        rand_val = np.random.random()
        # print("epsilon = ",epsilon)
        # print("rand_val = ",rand_val)
        if rand_val < epsilon:
            # print("YESS")
            action = random.randint(0, num_actions-1)
            # print("Action = ",action)
        else:
            # print("NOO")
            maxVal = float('-inf')
            for vals in Q[state]:
                if vals > maxVal:
                    maxVal = vals
            t=0
            best_actions = []
            # print("max val = ",maxVal)
            for vals in Q[state]:
                if vals == maxVal:
                    best_actions.append(t)
                t +=1
            # print("best_actions = ",best_actions)
            num_best_actions = len(best_actions)
            # print("num_best_actions = ",num_best_actions)
            action = best_actions[random.randint(0, num_best_actions-1)]
            # print("action = ",action)
        return action

    return get_action


def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

def random_walk_policy() -> Callable:
    
    def get_action(state: Tuple) -> int:
        if np.random.random() <= 0.5:
            action = 0
        else:
            action = 1
        return action

    return get_action





