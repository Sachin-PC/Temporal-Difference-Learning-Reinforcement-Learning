import gym
from typing import Optional, Callable, Tuple
from collections import defaultdict
import numpy as np
from tqdm import trange
from policy import create_epsilon_policy, random_walk_policy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float, maximum_steps
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)
    episode_steps = []
    steps_count = 0
    episode_steps.append(steps_count)
    per_episode_steps_limit = 1000
    min_steps = 20000000
    episode_ten_k = -1
    count = 0
    average_steps_sum = []
    steps_episode_num = []
    returns = np.zeros(num_episodes)
    for i in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env, policy)
        print("Episode len = ",len(episode))

        # print("episode len = ",len(episode))
        # print("episode = ",i," len = ",len(episode))
        # print("episode = ",i)
        G = 0
        # state_index_episode = defaultdict(int)
        # for t in range(len(episode)):
        #     state, _, _ = episode[t]
        #     if state_index_episode.get(state) is not None:
        #         state_index_episode.get(state).append(t)
        #     else:
        #         state_index_episode[state] = [t]

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma*G + reward
            # if is_state_first_appearance(state_index_episode,state,t):
            if (state, action) not in [(e[0], e[1]) for e in episode[:t]]:
                prev_num_state_samples = N[state][action]
                if prev_num_state_samples == 0: 
                    Q[state][action] = G
                else:
                    state_prev_average_reward = Q[state][action]
                    state_new_reward = (G + (prev_num_state_samples*state_prev_average_reward))/(prev_num_state_samples + 1)
                    Q[state][action] = state_new_reward
                N[state][action] = prev_num_state_samples + 1
        policy = create_epsilon_policy(Q, epsilon)
        returns[i] = G
        steps_count += len(episode)
        for sen in range(len(episode)):
            steps_episode_num.append(i)    
        if steps_count < maximum_steps:
            if len(episode) < min_steps:
                min_steps = len(episode)
            if i >= 200:
                average_steps_sum.append(i)
            episode_steps.append(steps_count)
        else:
            episode_ten_k = i+1
            break
    average_steps_count = np.mean(average_steps_sum)
    return returns, episode_steps, min_steps, episode_ten_k, average_steps_count, steps_episode_num[0:maximum_steps]
    # return returns



def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float, num_episodes, maximum_steps):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q,epsilon)
    episode_steps = []
    steps_count = 0
    episode_steps.append(steps_count)
    per_episode_steps_limit = 1000
    min_steps = 2000
    episode_ten_k = -1
    count = 0
    average_steps_sum = []
    steps_episode_num = []
    for episode_num in trange(num_episodes, desc="Episode"):
        state = env.reset()
        action = policy(state)
        for i in range(per_episode_steps_limit):
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            Q[state][action] = Q[state][action] + step_size*(reward + (gamma*Q[next_state][next_action]) - Q[state][action])
            state = next_state
            action = next_action
            steps_count += 1
            steps_episode_num.append(episode_num)
            if steps_count == maximum_steps:
                episode_ten_k = episode_num+1
                break
            if done:
                break
        i +=1
        if steps_count < maximum_steps:
            if i < min_steps:
                min_steps = i 
            if episode_num >= 200:
                average_steps_sum.append(i)
            episode_steps.append(steps_count)
        else:
            break
    average_steps_count = np.mean(average_steps_sum)
    # V = defaultdict(float)
    # for q_val in Q:
    #     V[q_val] = max(Q[q_val])
    # for i in range(10):
    #     for j in range(7):
    #         print(round(V[(i,j)],2)," ",end ="")
    #     print()
    return Q, episode_steps, min_steps, episode_ten_k, average_steps_count,steps_episode_num



def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    num_episodes: int,
    maximum_steps
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q,epsilon)
    n = num_steps
    episode_steps = []
    steps_count = 0
    episode_steps.append(steps_count)
    per_episode_steps_limit = 1000
    min_steps = 2000
    episode_ten_k = -1
    count = 0
    average_steps_sum = []
    steps_episode_num = []
    terminal_states = [env.goal_pos]
    for episode_num in trange(num_episodes, desc="Episode"):
        rewards_list = []
        states_list = []
        actions_list = []
        T = 3000
        for j in range(T+2):
            rewards_list.append(0)
            states_list.append((-1,-1))
            actions_list.append(-1)
        state = env.reset()
        action = policy(state)
        states_list[0] = state
        actions_list[0] = action
        for t in range(T):
            if t < T:
                next_state, reward, done, _ = env.step(action)
                # print("state = ",state," action = ",action," next_state = ",next_state,"reward = ",reward,"done = ",done)
                rewards_list[t+1] = reward
                states_list[t+1] = next_state
                # To count all steps taken
                steps_count += 1
                steps_episode_num.append(episode_num)
                if steps_count == maximum_steps:
                    episode_ten_k = episode_num+1
                if done:
                    T = t+1
                    # print("\nINSIDE TERMINAL STATE 1")
                if next_state in terminal_states:
                    # print("INSIDE TERMINAL STATE 2")
                    T = t+1
                else:
                    next_action = policy(next_state)
                    actions_list[t+1] = next_action
            state_estimate_t = t - n + 1
            if state_estimate_t >= 0:
                start_index = state_estimate_t + 1
                end_index = min(state_estimate_t + n, T)
                G = 0
                for i in range(start_index, end_index+1):
                    G += (gamma**(i - state_estimate_t - 1))*rewards_list[i]
                if state_estimate_t + n < T:
                    last_state = states_list[state_estimate_t + n]
                    last_state_action = actions_list[state_estimate_t + n]
                    G = G + (gamma**n)*Q[last_state][last_state_action]
                upadting_state = states_list[state_estimate_t]
                upadting_state_action = actions_list[state_estimate_t]
                Q[upadting_state][upadting_state_action] += step_size*(G - Q[upadting_state][upadting_state_action])
            if state_estimate_t == T - 1:
                break
            state = next_state
            action = next_action
        # no_of_steps += T
        if T < min_steps:
            min_steps = i 
        if episode_num >= 200:
            average_steps_sum.append(T)
        episode_steps.append(steps_count)
        if steps_count >= maximum_steps:
            break
        # episode_steps.append(no_of_steps)
    average_steps_count = np.mean(average_steps_sum)
    return Q, episode_steps, min_steps, episode_ten_k, average_steps_count, steps_episode_num[0:maximum_steps]


def nstep_td_prediction_original(
    env: gym.Env,
    Q,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    num_episodes: int,
    maximum_steps
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    V = defaultdict(float)
    for r in range(env.rows):
        for c in range(env.cols):
            V[(r,c)] = 0.0
    policy = create_epsilon_policy(Q,epsilon)
    n = num_steps
    episode_steps = []
    steps_count = 0
    episode_steps.append(steps_count)
    per_episode_steps_limit = 1000
    min_steps = 2000
    episode_ten_k = -1
    count = 0
    average_steps_sum = []
    steps_episode_num = []
    terminal_states = [env.goal_pos]
    for episode_num in trange(num_episodes, desc="Episode"):
        rewards_list = []
        states_list = []
        actions_list = []
        T = 3000
        for j in range(T+2):
            rewards_list.append(0)
            states_list.append((-1,-1))
            actions_list.append(-1)
        state = env.reset()
        states_list[0] = state
        for t in range(T):
            if t < T:
                action = policy(state)
                actions_list[0] = action
                next_state, reward, done, _ = env.step(action)
                # print("state = ",state," action = ",action," next_state = ",next_state,"reward = ",reward,"done = ",done)
                rewards_list[t+1] = reward
                states_list[t+1] = next_state
                # To count all steps taken
                steps_count += 1
                steps_episode_num.append(episode_num)
                if steps_count == maximum_steps:
                    episode_ten_k = episode_num+1
                if done:
                    T = t+1
                    # print("\nINSIDE TERMINAL STATE 1")
                if next_state in terminal_states:
                    # print("INSIDE TERMINAL STATE 2")
                    T = t+1
                # else:
                #     next_action = policy(next_state)
                #     actions_list[t+1] = next_action
            state_estimate_t = t - n + 1
            if state_estimate_t >= 0:
                G = sum(gamma**(i - state_estimate_t - 1) * rewards_list[i] for i in range(state_estimate_t + 1, min(state_estimate_t + n, t + 1))) + \
                    (gamma**min(n, t - state_estimate_t)) * V[states_list[t + 1]]
                # start_index = state_estimate_t + 1
                # end_index = min(state_estimate_t + n, T)
                # G = 0
                # for i in range(start_index, end_index+1):
                #     prev_G = G
                #     G += (gamma**(i - state_estimate_t - 1))*rewards_list[i]
                #     if state_estimate_t + n < T:
                #         last_state = states_list[state_estimate_t + n]
                #         # last_state_action = actions_list[state_estimate_t + n]
                #         G = G + (gamma**n)*V[last_state]
                #     if np.isnan(G):
                #         # print(V)
                #         print("gamma**n = ",gamma**n)
                #         print("last_state = ",last_state)
                #         print("V[last_state] = ",V[last_state])
                #         print("rewards_list[i] = ",rewards_list[i])

                upadting_state = states_list[state_estimate_t]
                # print("UPDATING STATE = ",upadting_state)
                # print("V[upadting_state] before = ",V[upadting_state])
                # print("g = ",G)
                # upadting_state_action = actions_list[state_estimate_t]
                V[upadting_state] += step_size*(G - V[upadting_state])
                if np.isnan(V[upadting_state]):
                    print("UPDATING STATE = ",upadting_state)
                    print("V[upadting_state] before = ",V[upadting_state])
                    print("g = ",G)


            if state_estimate_t == T - 1:
                break
            state = next_state
            # action = next_action
        # no_of_steps += T
        if T < min_steps:
            min_steps = i 
        if episode_num >= 200:
            average_steps_sum.append(T)
        episode_steps.append(steps_count)
        if steps_count >= maximum_steps:
            break
        # episode_steps.append(no_of_steps)
    average_steps_count = np.mean(average_steps_sum)
    return V


def nstep_td_prediction(
    env: gym.Env,
    Q,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    num_episodes: int,
    maximum_steps
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    V = defaultdict(float)
    policy = create_epsilon_policy(Q,epsilon)
    n = num_steps
    terminal_states = [env.goal_pos]
    for episode_num in trange(num_episodes, desc="Episode"):
        rewards_list = []
        states_list = []
        # actions_list = []
        T = 3000
        # for j in range(T+2):
        #     rewards_list.append(0)
        #     states_list.append((-1,-1))
        #     actions_list.append(-1)
        state = env.reset()
        states_list.append(state)
        rewards_list.append(0)
        t = 0
        for t in range(T):
            if t < T:
                action = policy(state)
                # actions_list.append(action)
                next_state, reward, done, _ = env.step(action)
                # print("state = ",state," action = ",action," next_state = ",next_state,"reward = ",reward,"done = ",done)
                rewards_list.append(reward)
                states_list.append(next_state)
                # To count all steps taken
                # steps_count += 1
                # steps_episode_num.append(episode_num)
                # if steps_count == maximum_steps:
                #     episode_ten_k = episode_num+1
                if done:
                    T = t+1
                    # print("\nINSIDE TERMINAL STATE 1")
                if next_state in terminal_states:
                    # print("INSIDE TERMINAL STATE 2")
                    T = t+1
                # else:
                #     next_action = policy(next_state)
                #     actions_list[t+1] = next_action
            state_estimate_t = t - n + 1
            if state_estimate_t >= 0:
                start_index = state_estimate_t + 1
                end_index = min(state_estimate_t + n, T)
                G = 0
                for i in range(start_index, end_index+1):
                    G += (gamma**(i - state_estimate_t - 1))*rewards_list[i]
                if state_estimate_t + n < T:
                    last_state = states_list[state_estimate_t + n]
                    # last_state_action = actions_list[state_estimate_t + n]
                    G += (gamma**n)*V[last_state]
                if np.isnan(G):
                    # print(V)
                    print("gamma**n = ",gamma**n)
                    print("last_state = ",last_state)
                    print("V[last_state] = ",V[last_state])
                    print("rewards_list[i] = ",rewards_list[i])

                upadting_state = states_list[state_estimate_t]
                V[upadting_state] += step_size * (G - V[upadting_state])
                if np.isnan(V[upadting_state]):
                    print("UPDATING STATE = ",upadting_state)
                    print("V[upadting_state] before = ",V[upadting_state])
                    print("g = ",G)


            if state_estimate_t == T - 1:
                break
            if done:
                break
            state = next_state
            # action = next_action
    return V

def get_next_state_expected_value(Q,state):
    prob_val = 1/len(Q[state])
    expected_val = 0
    for vals in Q[state]:
        expected_val += prob_val*vals
    return expected_val


def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    num_episodes: int,
    maximum_steps
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q,epsilon)
    episode_steps = []
    per_episode_steps_limit = 1000
    episode_ten_k = -1
    min_steps = 2000
    steps_count = 0
    average_steps_sum = []
    steps_episode_num = []
    episode_steps.append(steps_count)
    for episode_num in trange(num_episodes, desc="Episode"):
        state = env.reset()
        for i in range(per_episode_steps_limit):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            next_state_expectedvalue = get_next_state_expected_value(Q,next_state)
            Q[state][action] = Q[state][action] + step_size*(reward + (gamma*next_state_expectedvalue) - Q[state][action])
            state = next_state
            steps_count += 1
            steps_episode_num.append(episode_num)
            if steps_count == 8000:
                episode_ten_k = episode_num+1
            if done:
                break
        i +=1
        if i < min_steps:
            min_steps = i 
        if episode_num >= 200:
            average_steps_sum.append(i)
        episode_steps.append(steps_count)
        if steps_count >= maximum_steps:
            break

    average_steps_count = np.mean(average_steps_sum)

    return Q, episode_steps, min_steps, episode_ten_k, average_steps_count, steps_episode_num[0:maximum_steps]

def get_best_action(Q, state):
    maxVal = -99999999
    for vals in Q[state]:
        if vals > maxVal:
            maxVal = vals
    t=0
    best_actions = []
    for vals in Q[state]:
        if vals == maxVal:
            best_actions.append(t)
        t +=1
    num_best_actions = len(best_actions)
    action = best_actions[random.randint(0, num_best_actions-1)]
    return action

def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
    num_episodes: int,
    maximum_steps
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q,epsilon)
    episode_steps = []
    per_episode_steps_limit = 1000
    episode_ten_k = -1
    min_steps = 2000
    steps_count = 0
    average_steps_sum = []
    steps_episode_num = []
    episode_steps.append(steps_count)
    for episode_num in trange(num_episodes, desc="Episode"):
        state = env.reset()
        for i in range(per_episode_steps_limit):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            # print("state = ",state," next_state = ",next_state,"reward = ",reward,"done = ",done)
            next_action = get_best_action(Q,next_state)
            Q[state][action] = Q[state][action] + step_size*(reward + (gamma*Q[next_state][next_action]) - Q[state][action])
            state = next_state
            # action = next_action
            steps_count += 1
            steps_episode_num.append(episode_num)
            if steps_count == maximum_steps:
                episode_ten_k = episode_num+1
            if done:
                break
        i +=1
        if i < min_steps:
            min_steps = i 
        if episode_num >= 200:
            average_steps_sum.append(i)
        episode_steps.append(steps_count)
        if steps_count >= maximum_steps:
            break

    average_steps_count = np.mean(average_steps_sum)
    return Q, episode_steps, min_steps, episode_ten_k, average_steps_count, steps_episode_num[0:maximum_steps]


def td_prediction(env: gym.Env, gamma: float, alpha, episodes, n):
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    gamma = 1.0
    total_states = env.states + 2
    V = [0.5]*(env.states+2)
    V[0] = 0
    V[-1] = 0
    policy = random_walk_policy()
    num_steps = 1000
    episode_v_values = []
    episode_v_values.append([0.5,0.5,0.5,0.5,0.5])
    required_episodes = [0,9,999]
    for episode in trange(episodes, desc="Episode"):
        state = env.reset()
        for _ in range(num_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                V[state] += alpha * (reward + V[next_state] - V[state])
                break
            V[state] +=  alpha*(reward + (gamma*V[next_state]) - V[state])
            state = next_state
        if episode in required_episodes:
            v_copy = np.copy(V[1:-1])
            episode_v_values.append(v_copy)
    episode_v_values.append([0.17, 0.33, 0.5, 0.67, 0.83])
    return episode_v_values

def find_rms(V):
    state_V = np.array(V[1:-1])
    state_true_values = np.array([0.17, 0.33, 0.5, 0.67, 0.83])
    error_square = (state_V - state_true_values)**2
    rms_value = np.sqrt(np.mean(error_square))
    return rms_value

def td_prediction_emperical_error(env: gym.Env, gamma: float, alpha: float,  episodes, n=1):
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    gamma = 1.0
    total_states = env.states + 2
    V = [0.5]*(env.states+2)
    V[0] = 0
    V[-1] = 0
    policy = random_walk_policy()
    num_steps = 1000
    rms_values = np.zeros(episodes)
    for episode in trange(episodes, desc="Episode"):
        state = env.reset()         
        for _ in range(num_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                V[state] += alpha * (reward + V[next_state] - V[state])
                break
            V[state] +=  alpha*(reward + (gamma*V[next_state]) - V[state])
            state = next_state
        rms_value = find_rms(V)
        rms_values[episode] = rms_value
    return rms_values

def mc_prediction_emperical_error(env: gym.Env, gamma: float, alpha: float,  episodes, n=1):
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    gamma = 1.0
    total_states = env.states + 2
    V = [0.5]*(env.states+2)
    V[0] = 0
    V[-1] = 0
    policy = random_walk_policy()
    num_steps = 1000
    rms_values = np.zeros(episodes)
    for e in trange(episodes, desc="Episode"):
        episode = generate_montecarlo_episode(env, policy, es=True)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma*G + reward
            V[state] += alpha*(G - V[state])
        rms_value = find_rms(V)
        rms_values[e] = rms_value
    return rms_values




def learning_targets(
    V, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO
    targets = np.zeros(len(episodes))
    i = 0
    if n == None:
        #Monte Carlo
        for episode in episodes:
            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = gamma*G + reward
            targets[i] = G
            i +=1
    elif n == 1:
        #TD(0)
        for episode in episodes:
            state, action, reward = episode[0]
            next_state, next_action, next_reward = episode[1]
            # print("state = ",state)
            # print("V state = ",V[state])
            target = reward + gamma*V[next_state]
            targets[i] = target
            i += 1 
    
    elif n > 1:
        print(V)
        #TD(n)
        for episode in episodes:
            target = 0
            for j in range(len(episode)):
                if j < n:
                    state, action, reward = episode[j]
                    target += (gamma**j)*reward
                else:
                    break
                state, action, reward = episode[j]
                target += (gamma**j)*V[state]
            targets[i] = target
            i += 1 
    return targets

def td_prediction_windy_grid_env(env: gym.Env, Q, gamma: float, episodes, n=1, epsilon = 0.1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    V = defaultdict(float)
    policy = create_epsilon_policy(Q,epsilon)
    alpha = 0.5
    num_steps = 2000
    for episode in trange(episodes, desc="Episode"):
        state = env.reset()
        for _ in range(num_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                V[state] += alpha * (reward + V[next_state] - V[state])
                break
            V[state] +=  alpha*(reward + (gamma*V[next_state]) - V[state])
            state = next_state
    return V


def generate_episode(env: gym.Env, policy: Callable):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    count =0
    while True: 
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        if count == 10000:
            break
        state = next_state
        count +=1

    return episode

def generate_montecarlo_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    count =0
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            # print("done")
            break
        if count == 1000:
            break
        state = next_state
        count +=1

    return episode


def on_policy_mc_evaluation(
    env: gym.Env,
    Q,
    num_episodes: int,
    gamma: float,
    epsilon,
) -> defaultdict:
    """On-policy Monte Carlo policy evaluation. First visits will be used.

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP

    Returns:
        V (defaultdict): The values for each state. V[state] = value.
    """
    # We use defaultdicts here for both V and N for convenience. The states will be the keys.
    V = defaultdict(float)
    N = defaultdict(int)
    policy = create_epsilon_policy(Q,epsilon)
    state_space = [
            (x, y) for x in range(0,env.rows) for y in range(0,env.cols)
        ]
    # print("state_space shape = ",state_space.shape)

    for _ in trange(num_episodes, desc="Episode"):
        episode = generate_montecarlo_episode(env, policy, es=True)
        # print("Episode = ",episode)
        G = 0
        state_index_episode = defaultdict(int)
        for t in range(len(episode)):
            state, _, _ = episode[t]
            if state_index_episode.get(state) is not None:
                state_index_episode.get(state).append(t)
            else:
                state_index_episode[state] = [t]

        for t in range(len(episode) - 1, -1, -1):
            # TODO Q3a
            state, action, reward = episode[t]
            G = gamma*G + reward
            if (state, action) not in [(e[0], e[1]) for e in episode[:t]]:
                prev_num_state_samples = N[state]
                if prev_num_state_samples == 0: 
                    V[state] = G
                else:
                    state_prev_average_reward = V[state]
                    state_new_reward = (G + (prev_num_state_samples*state_prev_average_reward))/(prev_num_state_samples + 1)
                    V[state] = state_new_reward
                N[state] = prev_num_state_samples + 1
    return V

