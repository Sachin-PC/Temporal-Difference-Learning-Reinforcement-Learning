from env import *
from kingsMovesEnv import *
from kingsMovesNineActionsEnv import *
from stochasticWindEnv import *
from randomWalkEnv import *
# from algorithms import *
from algorithms import *
from policy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from typing import Any

def main():
    # q3_a()
    # q3_b()
    q4_b()
    # q4_c1()
    # q4_c2()
    # q4_d()
    # q5()
    # q5New()
    # q5_c()

def q3_a():
    env = RandomWalkEnv()
    gamma = 1
    alpha = 0.1
    episodes = 1000
    n = 1
    episode_v_values = td_prediction(env, gamma, alpha, episodes, n)
    plot_label = ["0", "1","10","100","True Value"]
    linestyles = ["solid","solid","solid","solid","solid"]
    plot_color = ["black","red","green","blue","black"]
    q3_a_plot(episode_v_values,plot_label,linestyles,plot_color)

def q3_b():
    env = RandomWalkEnv()
    gamma = 1
    alphas = [0.05, 0.1, 0.15]
    td_plot_label = ["alpha = 0.05","alpha = 0.1","alpha = 0.15"]
    td_linestyles = ["solid","dotted","dashed"]
    td_color = ["blue","blue","blue"]
    episodes = 100
    n = 1
    trials = 100
    # td_rms_values_all_trials = np.zeros(shape=(trials,episodes))
    td_rms_errors = []
    for alpha in alphas:
        td_rms_values_all_trials = np.zeros(shape=(trials,episodes))
        for trial in range(trials):
            rms_values = td_prediction_emperical_error(env, gamma, alpha,  episodes, n)
            td_rms_values_all_trials[trial] = rms_values
        average_rms_error = np.mean(td_rms_values_all_trials,axis = 0)
        td_rms_errors.append(average_rms_error)

    alphas = [0.01, 0.02, 0.03, 0.04]
    mc_plot_label = ["alpha = 0.01","alpha = 0.02","alpha = 0.03","alpha = 0.04"]
    mc_linestyles = ["solid","dotted","dashed","dashdot"]
    mc_color = ["red","red","red","Red"]
    mc_rms_errors = []
    for alpha in alphas:
        mc_rms_values_all_trials = np.zeros(shape=(trials,episodes))
        for trial in range(trials):
            rms_values = mc_prediction_emperical_error(env, gamma, alpha,  episodes, n)
            mc_rms_values_all_trials[trial] = rms_values
        average_rms_error = np.mean(mc_rms_values_all_trials,axis = 0)
        mc_rms_errors.append(average_rms_error)
    
    rms_errors = td_rms_errors + mc_rms_errors
    labels = td_plot_label + mc_plot_label
    linestyles = td_linestyles + mc_linestyles
    color = td_color + mc_color
    q3_a_plot(rms_errors,labels,linestyles,color)

def q3_a_plot(V_list,plot_label,linestyles,plot_color):
    # plot_color = ["blue","blue","blue"]
    # plot_label = []
    # print()
    for i in range(len(V_list)):
        plt.plot(V_list[i],color = plot_color[i], label = plot_label[i], linestyle = linestyles[i])

    plt.xlabel("State")
    plt.ylabel("Estimated Value")
    plt.title("Estimated Value over multiple episodes")
    plt.legend(loc='lower right')
    plt.show()

def q3_b_plot(rms_errors,plot_label,linestyles,plot_color):
    # plot_color = ["blue","blue","blue"]
    # plot_label = []
    # print()
    for i in range(len(rms_errors)):
        plt.plot(rms_errors[i],color = plot_color[i], label = plot_label[i], linestyle = linestyles[i])

    plt.xlabel("Episodes")
    plt.ylabel("RMS Value")
    plt.title("Emperical RMS error averaged over states")
    plt.legend(loc='lower right')
    plt.show()

def get_confidence_band(steps_data, trials):
    average_episodes = np.mean(steps_data, axis=0)
    episode_steps_std = np.std(average_episodes, axis=0)
    confindence_band = ((episode_steps_std)/(math.sqrt(trials)))*1.96
    return average_episodes, confindence_band
    # plt.plot(steps_episode_num,color = "green", label = "sarsa")
    # plt.plot(average_return,color = colours[i], label = label_string )


def q4_b():
    env = WindyGridWorldEnv()
    epsilon = 0.1
    gamma = 1
    num_steps = 4
    step_size = 0.5
    num_episodes = 5000
    episodes = np.ones(num_episodes+1, dtype=np.float32)
    for i in range(num_episodes+1):
        episodes[i] = i
    plot_data = []
    plot_label = []
    plot_color = []
    confidence_bands = []
    title = "Windy GridWorldEnv Avg Steps/Episode"
    trials = 10 
    maximum_steps = 10000
    average_episode_steps = np.zeros((trials,maximum_steps))
    steps_deg = np.ones(maximum_steps, dtype=np.int)
    for i in range(maximum_steps):
        steps_deg[i] = i

    # # monte carlo
    # for trial in range(trials):
    #     mc_control_return, mc_episode_steps, min_steps, episode_ten_k, average_steps_count, mc_steps_episode_num = on_policy_mc_control_epsilon_soft(env, num_episodes, gamma, epsilon, maximum_steps)
    #     print(len(mc_steps_episode_num))
    #     average_episode_steps[trial] = mc_steps_episode_num
    #     print("For MC CONTROL Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # mc_control_average_episodes, mc_control_confidence_band = get_confidence_band(average_episode_steps, trials)
    # plot_data.append(mc_control_average_episodes)
    # confidence_bands.append(mc_control_confidence_band)
    # plot_label.append("mc control")
    # plot_color.append("black")

    #sarsa
    for trial in range(trials):
        sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, sarsa_steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    sarsa_average_episodes, sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(sarsa_average_episodes)
    confidence_bands.append(sarsa_confidence_band)
    plot_label.append("SARSA")
    plot_color.append("green")


    num_steps = 4

    #nstep_sarsa
    for trial in range(trials):
        nstepsarsa_Q, nsteps_sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, nstep_sarsa_steps_episode_num = nstep_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = nstep_sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    nstep_sarsa_average_episodes, nstep_sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(nstep_sarsa_average_episodes)
    confidence_bands.append(nstep_sarsa_confidence_band)
    plot_label.append("nstep SARSA")
    plot_color.append("orange")


    #expected SARSA
    for trial in range(trials):
        expectedSarsa_Q, expectedsarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, exp_sarsa_steps_episode_num = exp_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = exp_sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    exp_sarsa_average_episodes, exp_sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(exp_sarsa_average_episodes)
    confidence_bands.append(exp_sarsa_confidence_band)
    plot_label.append("Expected SARSA")
    plot_color.append("red")


    #Q learning
    for trial in range(trials):
        qlearning_Q, qlearning_episode_steps, min_steps, episode_ten_k, average_steps_count, qlearning_steps_episode_num = q_learning(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = qlearning_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    qlearning_average_episodes, qlearning_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(qlearning_average_episodes)
    confidence_bands.append(qlearning_confidence_band)
    plot_label.append("Q learning")
    plot_color.append("blue")

    create_plot_q4_confidenceband(plot_data,confidence_bands,plot_label,plot_color, steps_deg,title)
    


    # 
    # print("For MC CONTROL Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # plot_data.append(mc_episode_steps)
    # plot_label.append("mc control")
    # plot_color.append("black")
    # sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes)
    # print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # plot_data.append(sarsa_episode_steps)
    # plot_label.append("sarsa")
    # plot_color.append("green")
    # sarsa_episode_steps[]
    # print(np.array(sarsa_episode_steps).shape)
    # print(np.array(stesp_episode_num).shape)
    # print(sarsa_episode_steps[-1])
    # qlearning_Q, qlearning_episode_steps, min_steps, episode_ten_k, average_steps_count = q_learning(env, num_steps, gamma, epsilon, step_size, num_episodes)
    # print("For Q Learning Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # plot_data.append(qlearning_episode_steps)
    # plot_label.append("qlearning")
    # plot_color.append("blue")
    # expectedSarsa_Q, expectedsarsa_episode_steps, min_steps, episode_ten_k, average_steps_count = exp_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes)
    # print("For Expected SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # plot_data.append(expectedsarsa_episode_steps)
    # plot_label.append("expected sarsa")
    # plot_color.append("red")
    # num_steps = 4
    # qlearning_Q, nsteps_sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count = nstep_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes)
    # print("For n step SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # plot_data.append(nsteps_sarsa_episode_steps)
    # plot_label.append("nsteps sarsa")
    # plot_color.append("orange")

    # plt.xlabel("time steps")
    # plt.ylabel("Episodes")
    # plt.title("Average Steps/Episode")
    # plt.legend(loc='lower right')
    # plt.show()
    # print(steps_episode_num)

    # create_plot_q4(episodes,plot_data,plot_label,plot_color)

def create_plot_q4_confidenceband(plot_data,confidence_bands,plot_label,plot_color, steps_deg, title):

    for i in range(len(plot_data)):
        plt.plot(steps_deg, plot_data[i],color = plot_color[i], label = plot_label[i])
        plt.fill_between(steps_deg,plot_data[i] - confidence_bands[i],plot_data[i]+confidence_bands[i],alpha=0.5,color = plot_color[i])
    plt.xlabel("time steps")
    plt.ylabel("Episodes")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def q4_c1():

    env = KingsWindyGridWorldEnv()
    epsilon = 0.1
    gamma = 1
    num_steps = 1
    step_size = 0.5
    num_episodes = 6000
    episodes = np.ones(num_episodes+1, dtype=np.float32)
    for i in range(num_episodes+1):
        episodes[i] = i
    plot_data = []
    plot_label = []
    plot_color = []
    confidence_bands = []
    title = "Kings move Windy GridWorld With 8 actions Avg Steps/Episode"
    trials = 10 
    maximum_steps = 10000
    average_episode_steps = np.zeros((trials,maximum_steps))
    steps_deg = np.ones(maximum_steps, dtype=np.int)
    for i in range(maximum_steps):
        steps_deg[i] = i
    #sarsa
    for trial in range(trials):
        sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, sarsa_steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    sarsa_average_episodes, sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(sarsa_average_episodes)
    confidence_bands.append(sarsa_confidence_band)
    plot_label.append("SARSA")
    plot_color.append("green")

    # num_steps = 4
    # #nstep_sarsa
    # for trial in range(trials):
    #     nstepsarsa_Q, nsteps_sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, nstep_sarsa_steps_episode_num = nstep_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
    #     average_episode_steps[trial] = nstep_sarsa_steps_episode_num
    #     print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # nstep_sarsa_average_episodes, nstep_sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    # plot_data.append(nstep_sarsa_average_episodes)
    # confidence_bands.append(nstep_sarsa_confidence_band)
    # plot_label.append("nstep SARSA")
    # plot_color.append("orange")


    #expected SARSA
    for trial in range(trials):
        expectedSarsa_Q, expectedsarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, exp_sarsa_steps_episode_num = exp_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = exp_sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    exp_sarsa_average_episodes, exp_sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(exp_sarsa_average_episodes)
    confidence_bands.append(exp_sarsa_confidence_band)
    plot_label.append("Expected SARSA")
    plot_color.append("red")

    create_plot_q4_confidenceband(plot_data,confidence_bands,plot_label,plot_color, steps_deg, title)

def q4_c2():

    env = KingsWindyGridWorldNineMovesEnv()
    epsilon = 0.1
    gamma = 1
    num_steps = 1
    step_size = 0.5
    num_episodes = 6000
    episodes = np.ones(num_episodes+1, dtype=np.float32)
    for i in range(num_episodes+1):
        episodes[i] = i
    plot_data = []
    plot_label = []
    plot_color = []
    confidence_bands = []
    title = "Kings move Windy GridWorld With 9 actions Avg Steps/Episode"
    trials = 10 
    maximum_steps = 10000
    average_episode_steps = np.zeros((trials,maximum_steps))
    steps_deg = np.ones(maximum_steps, dtype=np.int)
    for i in range(maximum_steps):
        steps_deg[i] = i
    #sarsa
    for trial in range(trials):
        sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, sarsa_steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    sarsa_average_episodes, sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(sarsa_average_episodes)
    confidence_bands.append(sarsa_confidence_band)
    plot_label.append("SARSA")
    plot_color.append("green")

    # num_steps = 4
    # #nstep_sarsa
    # for trial in range(trials):
    #     nstepsarsa_Q, nsteps_sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, nstep_sarsa_steps_episode_num = nstep_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
    #     average_episode_steps[trial] = nstep_sarsa_steps_episode_num
    #     print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # nstep_sarsa_average_episodes, nstep_sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    # plot_data.append(nstep_sarsa_average_episodes)
    # confidence_bands.append(nstep_sarsa_confidence_band)
    # plot_label.append("nstep SARSA")
    # plot_color.append("orange")


    #expected SARSA
    for trial in range(trials):
        expectedSarsa_Q, expectedsarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, exp_sarsa_steps_episode_num = exp_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = exp_sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    exp_sarsa_average_episodes, exp_sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(exp_sarsa_average_episodes)
    confidence_bands.append(exp_sarsa_confidence_band)
    plot_label.append("Expected SARSA")
    plot_color.append("red")

    create_plot_q4_confidenceband(plot_data,confidence_bands,plot_label,plot_color, steps_deg, title)

def q4_d():

    env = StochasticWindyGridWorldEnv()
    epsilon = 0.1
    gamma = 1
    num_steps = 1
    step_size = 0.5
    num_episodes = 60000
    episodes = np.ones(num_episodes+1, dtype=np.float32)
    for i in range(num_episodes+1):
        episodes[i] = i
    plot_data = []
    plot_label = []
    plot_color = []
    confidence_bands = []
    title = "Kings move Windy GridWorld Avg Steps/Episode"
    trials = 10 
    maximum_steps = 100000
    average_episode_steps = np.zeros((trials,maximum_steps))
    steps_deg = np.ones(maximum_steps, dtype=np.int)
    for i in range(maximum_steps):
        steps_deg[i] = i
    #sarsa
    for trial in range(trials):
        sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, sarsa_steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    sarsa_average_episodes, sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(sarsa_average_episodes)
    confidence_bands.append(sarsa_confidence_band)
    plot_label.append("SARSA")
    plot_color.append("green")

    # num_steps = 4
    # #nstep_sarsa
    # for trial in range(trials):
    #     nstepsarsa_Q, nsteps_sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, nstep_sarsa_steps_episode_num = nstep_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
    #     average_episode_steps[trial] = nstep_sarsa_steps_episode_num
    #     print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    # nstep_sarsa_average_episodes, nstep_sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    # plot_data.append(nstep_sarsa_average_episodes)
    # confidence_bands.append(nstep_sarsa_confidence_band)
    # plot_label.append("nstep SARSA")
    # plot_color.append("orange")


    #expected SARSA
    for trial in range(trials):
        expectedSarsa_Q, expectedsarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, exp_sarsa_steps_episode_num = exp_sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
        average_episode_steps[trial] = exp_sarsa_steps_episode_num
        print("For SARSA Algorithm:\n1. Min_steps taken by an episode = ",min_steps,"\n2. Episodes to reach 8k = ",episode_ten_k,"\n3. Average steps count after 200 episodes = ",average_steps_count)
    exp_sarsa_average_episodes, exp_sarsa_confidence_band = get_confidence_band(average_episode_steps, trials)
    plot_data.append(exp_sarsa_average_episodes)
    confidence_bands.append(exp_sarsa_confidence_band)
    plot_label.append("Expected SARSA")
    plot_color.append("red")

    create_plot_q4_confidenceband(plot_data,confidence_bands,plot_label,plot_color, steps_deg, title)

def q5():
    env = WindyGridWorldEnv()
    epsilon = 0.1
    gamma = 1
    num_steps = 1
    training_episodes = [1,20,50]
    step_size = 0.5
    num_episodes = 300
    maximum_steps = 10000
    sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, sarsa_steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
    n = 1
    td0_V = []
    tdn_V = []
    mc_V = []
    for episodes in training_episodes:
        V_val = td_prediction_windy_grid_env(env, sarsa_Q, gamma, episodes, n, epsilon)
        td0_V.append(V_val)
        mc_V_Val = on_policy_mc_evaluation(env,sarsa_Q,episodes,gamma,epsilon)
        mc_V.append(mc_V_Val)
        num_steps = 4
        tdn_V_val = nstep_td_prediction(env,sarsa_Q,num_steps,gamma,epsilon,step_size,num_episodes,maximum_steps)
        tdn_V.append(tdn_V_val)
    total_evaluation_episodes = 500
    evaluation_episodes = []
    policy = create_epsilon_policy(sarsa_Q,epsilon)
    for i in range(total_evaluation_episodes):
        episode = generate_episode(env, policy)
        evaluation_episodes.append(episode)
    td_evaluation_targets = []
    mc_evaluation_targets = []
    td_n_evaluation_targets = []
    for V in td0_V:
        td0_targets = learning_targets(V, gamma, evaluation_episodes, n)
        td_evaluation_targets.append(td0_targets)
        mc_targets = learning_targets(V, gamma, evaluation_episodes, None)
        mc_evaluation_targets.append(mc_targets)
    n = 4
    for V in tdn_V:
        tdn_targets = learning_targets(V, gamma, evaluation_episodes, n)
        td_n_evaluation_targets.append(tdn_targets)

    i = 0
    for evaluation in td_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of TD(0) Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i +=1

    i = 0
    for evaluation in mc_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of MC Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i+=1

    i = 0
    for evaluation in td_n_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of TD(n) Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i+= 1

def q5_c():
    env = WindyGridWorldEnv()
    epsilon = 0.1
    gamma = 1
    num_steps = 1
    training_episodes = [1,20,50]
    step_size = 0.5
    num_episodes = 300
    maximum_steps = 10000
    # sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, sarsa_steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
    n = 1
    td0_V = []
    tdn_V = []
    mc_V = []
    tdn_V_val = defaultdict(float)
    for episodes in training_episodes:
        sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, sarsa_steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, episodes, maximum_steps)
        V_val = td_prediction_windy_grid_env(env, sarsa_Q, gamma, episodes, n, epsilon)
        td0_V.append(V_val)
        mc_V_Val = on_policy_mc_evaluation(env,sarsa_Q,episodes,gamma,epsilon)
        mc_V.append(mc_V_Val)
        num_steps = 4
        tdn_V_val = nstep_td_prediction(env,sarsa_Q,num_steps,gamma,epsilon,step_size,num_episodes,maximum_steps)
        # tdn_V_val = td_n_prediction(env,tdn_V_val,num_steps,gamma,step_size,num_episodes,maximum_steps)
        tdn_V.append(tdn_V_val)
    total_evaluation_episodes = 500
    evaluation_episodes = []
    policy = create_epsilon_policy(sarsa_Q,epsilon)
    for i in range(total_evaluation_episodes):
        episode = generate_episode(env, policy)
        evaluation_episodes.append(episode)
    td_evaluation_targets = []
    mc_evaluation_targets = []
    td_n_evaluation_targets = []
    for V in td0_V:
        td0_targets = learning_targets(V, gamma, evaluation_episodes, n)
        td_evaluation_targets.append(td0_targets)
        mc_targets = learning_targets(V, gamma, evaluation_episodes, None)
        mc_evaluation_targets.append(mc_targets)
    n = 4
    for V in tdn_V:
        tdn_targets = learning_targets(V, gamma, evaluation_episodes, n)
        td_n_evaluation_targets.append(tdn_targets)

    i = 0
    for evaluation in td_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of TD(0) Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i +=1

    i = 0
    for evaluation in mc_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of MC Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i+=1

    i = 0
    for evaluation in td_n_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of TD(n) Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i+= 1


def q5New():
    env = WindyGridWorldEnv()
    epsilon = 0.1
    gamma = 1
    num_steps = 1
    training_episodes = [1,20,50]
    step_size = 0.5
    num_episodes = 300
    maximum_steps = 10000
    sarsa_Q, sarsa_episode_steps, min_steps, episode_ten_k, average_steps_count, sarsa_steps_episode_num = sarsa(env, num_steps, gamma, epsilon, step_size, num_episodes, maximum_steps)
    n = 1
    td0_V = []
    tdn_V = []
    mc_V = []
    for episodes in training_episodes:
        V_val = td_prediction_windy_grid_env(env, sarsa_Q, gamma, episodes, n, epsilon)
        td0_V.append(V_val)
        mc_V_Val = on_policy_mc_evaluation(env,sarsa_Q,episodes,gamma,epsilon)
        mc_V.append(mc_V_Val)
    
    policy = create_epsilon_policy(sarsa_Q,epsilon)
    nstep_training_episodes = []
    V = defaultdict(float)
    n = 40
    step_size = 0.5
    for i in range(50):
        states = []
        rewards = []
        episode = generate_episode(env, policy)
        nstep_training_episodes.append(episode)
        for j in range(len(episode)):
            states.append(episode[j][0])
            rewards.append(episode[j][2])
            # print("episode j = ",episode[j])
            # print("states = ",states)
            # print("rewards = ",rewards)
        V = n_step_td_prediction(states, rewards, V, n, step_size, gamma)
        if i in [0,19,49]:
            tdn_V.append(V)
    print(V)

    # num_steps = 4
    # tdn_V_val = nstep_td_prediction(env,sarsa_Q,num_steps,gamma,epsilon,step_size,num_episodes,maximum_steps)
    # tdn_V.append(tdn_V_val)

    total_evaluation_episodes = 500
    evaluation_episodes = []
    policy = create_epsilon_policy(sarsa_Q,epsilon)
    for i in range(total_evaluation_episodes):
        episode = generate_episode(env, policy)
        evaluation_episodes.append(episode)
    td_evaluation_targets = []
    mc_evaluation_targets = []
    td_n_evaluation_targets = []
    n = 1
    for V in td0_V:
        print("V = ")
        td0_targets = learning_targets(V, gamma, evaluation_episodes, n)
        td_evaluation_targets.append(td0_targets)
        mc_targets = learning_targets(V, gamma, evaluation_episodes, None)
        mc_evaluation_targets.append(mc_targets)
    n = 40
    for V in tdn_V:
        tdn_targets = learning_targets(V, gamma, evaluation_episodes, n)
        td_n_evaluation_targets.append(tdn_targets)

    i = 0
    for evaluation in td_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of TD(0) Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i +=1

    i = 0
    for evaluation in mc_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of MC Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i+=1

    i = 0
    for evaluation in td_n_evaluation_targets:
        print("evaluation = ",evaluation)
        plt.hist(evaluation, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Learning Targets')
        plt.ylabel('Frequency')
        title = 'Distribution of TD(n) Learning Targets for State S and trainig episodes'+str(training_episodes[i])
        plt.title(title)
        plt.show()
        i+= 1


def create_plot_q4(episodes,plot_data,plot_label,plot_color):
    for i in range(len(plot_data)):
        plt.plot(plot_data[i],episodes,color = plot_color[i], label = plot_label[i])

    plt.xlabel("time steps")
    plt.ylabel("Episodes")
    plt.title("Average Steps/Episode")
    plt.legend(loc='lower right')
    plt.show()

def create_plot_q3(states,plot_data,plot_label,plot_color):
    for i in range(len(plot_data)):
        plt.plot(states,plot_data[i],color = plot_color[i], label = plot_label[i])

    plt.xlabel("States")
    plt.ylabel("Expected Value")
    plt.title("Expected value using TD(0)")
    plt.legend(loc='lower right')
    plt.show()

def create_plot_q3_b(plot_data,plot_label,plot_color):
    for i in range(len(plot_data)):
        plt.plot(plot_data[i],color = plot_color[i], label = plot_label[i])

    plt.xlabel("Episodes")
    plt.ylabel("RMS Value")
    plt.title("Emperical RMS error averaged over states")
    plt.legend(loc='lower right')
    plt.show()









if __name__ == "__main__":
    main()