# Authors: A. Borromini, J. Grassi
# Date: 29_08_2023

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.optimize import linear_sum_assignment
from Environment import Environment
from Learners import Learner, UCBLearner, TSLearner
from Utils import simulate_episode, test_seed, greedy_algorithm, hungarian_algorithm, get_reward, clairvoyant


n_arms = 30
edge_rate=0.1
graph_structure = np.random.binomial(1, edge_rate, (n_arms, n_arms))
graph_probabilities = np.random.uniform(0.01, 0.99, (n_arms, n_arms)) * graph_structure


node_classes = 3
product_classes = 3
products_per_class = 3
T = 365

means = np.random.uniform(1, 20, size=(3,3))
std_dev = np.random.uniform(0, 1, size=(3,3))
rewards_parameters = (means, std_dev)
customer_assignments = np.random.choice([0,1,2], size=30)
n_exp = 25


"""ESTIMATING EDGE PROBABILITIES WITH UCB"""
def UCB_Generate_Probability_Estimates(p, n_arms=30, T = 365, n_experiments=10):

    experimentS_means_at_each_round = np.empty((n_experiments, T, n_arms))

    for e in tqdm(range(0, n_experiments)):
        # Initialize environment
        ucb_env = Environment(probabilities=p)
        # Initialize learner
        ucb_learner = UCBLearner(n_arms)

        for t in range(0, T):
            # SW-UCB Learner
            pulled_arm = ucb_learner.pull_arm()
            reward = ucb_env.round(pulled_arm)
            ucb_learner.update(pulled_arm, reward)

            # At each round memorize a copy of the means of each arm
            expected_rew = ucb_learner.expectations()
            experimentS_means_at_each_round[e, t] = expected_rew.copy()

    return experimentS_means_at_each_round

UCB_estimated_graph_probabilities = []

for index in range(len(graph_probabilities)):
    print("Estimating Activation Probabilities for Arm", index, "with UCB")
    arm_probabilities = UCB_Generate_Probability_Estimates(graph_probabilities[index])
    arm_probabilities = np.mean(arm_probabilities, axis=0)
    UCB_estimated_graph_probabilities.append(arm_probabilities)

UCB_estimated_graph_probabilities = np.array(UCB_estimated_graph_probabilities)
UCB_estimated_graph_probabilities = np.transpose(UCB_estimated_graph_probabilities, (1, 0, 2))


"""ESTIMATING EDGE PROBABILITIES WITH TS"""
def TS_Generate_Probability_Estimates(p, n_arms=30, T = 365, n_experiments=10):

    experimentS_means_at_each_round = np.empty((n_experiments, T, n_arms))

    for e in tqdm(range(0, n_experiments)):
        # Initialize environment
        ts_env = Environment(probabilities=p)
        # Initialize learner
        ts_learner = TSLearner(n_arms)

        for t in range(0, T):
            # TS-UCB Learner
            pulled_arm = ts_learner.pull_arm()
            reward = ts_env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            # At each round memorize a copy of the means of each arm
            expected_rew = ts_learner.expectations()
            experimentS_means_at_each_round[e, t] = expected_rew.copy()

    return experimentS_means_at_each_round

TS_estimated_graph_probabilities = []

for index in range(len(graph_probabilities)):
    print("Estimating Activation Probabilities for Arm", index, "with TS")
    arm_probabilities = TS_Generate_Probability_Estimates(graph_probabilities[index])
    arm_probabilities = np.mean(arm_probabilities, axis=0)
    TS_estimated_graph_probabilities.append(arm_probabilities)

TS_estimated_graph_probabilities = np.array(TS_estimated_graph_probabilities)
TS_estimated_graph_probabilities = np.transpose(TS_estimated_graph_probabilities, (1, 0, 2))

new_TS_estimated_graph_probabilities = []
for table in TS_estimated_graph_probabilities:
    cleaned_table = table*graph_structure
    new_TS_estimated_graph_probabilities.append(cleaned_table)
TS_estimated_graph_probabilities = new_TS_estimated_graph_probabilities


"""Instantaneous Rewards and Regrets for Edge Activation Probability Estimation"""

repeated_array = np.tile(graph_probabilities, (365, 1, 1))
original_probabilities = repeated_array.reshape((365, 30, 30))

UCB_nodes_probabilities = []
for i in range(len(UCB_estimated_graph_probabilities)):
    nodes_only = UCB_estimated_graph_probabilities[i]*graph_structure
    UCB_nodes_probabilities.append(nodes_only)
UCB_regret = np.sum((original_probabilities - UCB_nodes_probabilities), axis=(1, 2))

TS_nodes_probabilities = []
for i in range(len(TS_estimated_graph_probabilities)):
    nodes_only = TS_estimated_graph_probabilities[i]*graph_structure
    TS_nodes_probabilities.append(nodes_only)
TS_regret = np.sum((original_probabilities - TS_nodes_probabilities), axis=(1, 2))

reward_UCB = [np.sum(i) for i in UCB_nodes_probabilities]
reward_TS = [np.sum(i) for i in TS_nodes_probabilities]
optimum = [np.sum(graph_probabilities) for i in range(T)]

time_periods = range(1, 366)

# Plot the two lists
plt.figure(figsize=(10, 6))
plt.plot(time_periods, reward_UCB, color='blue', linestyle='-', label='UCB')
plt.plot(time_periods, reward_TS, color='red', linestyle='-', label='TS')
plt.plot(time_periods, optimum, color='lightgreen', linestyle='-', label='TS', linewidth=4)
plt.xlabel('Time')
plt.ylabel('Instantaneous Reward')
plt.title('Instantaneous Reward of UCB and TS probability estimations')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(time_periods, UCB_regret, color='blue', linestyle='-', label='UCB')
plt.plot(time_periods, TS_regret, color='red', linestyle='-', label='TS')
plt.xlabel('Time')
plt.ylabel('Instantaneous Regret')
plt.title('Instantaneous Regret of UCB and TS probability estimations')
plt.legend()
plt.grid()
plt.show()


"""Cumulative Regrets in probability estimation """

UCB_cumulative_regret = np.cumsum(UCB_regret)
TS_cumulative_regret = np.cumsum(TS_regret)

# Create the cumulative regret plot
plt.figure(figsize=(10, 6))
plt.plot(UCB_cumulative_regret, label='UCB Cumulative Regret', color="blue")
plt.plot(TS_cumulative_regret, label='TS Cumulative Regret', color = "red")
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret Comparison')
plt.legend()
plt.grid()
plt.show()


"""COMPUTING REWARDS WITH UCB ESTIMATES"""

n_exp = 5

UCB_mean_rewards_per_round = []
UCB_std_dev_rewards_per_round = []
for table in tqdm(range(T)):
    table = UCB_estimated_graph_probabilities[table]
    clairvoyant_output =  clairvoyant(table, graph_probabilities, customer_assignments, rewards_parameters, real_reward_parameters=rewards_parameters, n_exp=n_exp)
    UCB_mean_rewards_per_round.append(clairvoyant_output[0])
    UCB_std_dev_rewards_per_round.append(clairvoyant_output[1])


"""COMPUTING REWARDS WITH TS ESTIMATES"""

TS_mean_rewards_per_round = []
TS_std_dev_rewards_per_round = []
for table in tqdm(range(T)):
    table = TS_estimated_graph_probabilities[table]
    clairvoyant_output =  clairvoyant(table, graph_probabilities, customer_assignments, rewards_parameters, real_reward_parameters=rewards_parameters, n_exp=n_exp)
    TS_mean_rewards_per_round.append(clairvoyant_output[0])
    TS_std_dev_rewards_per_round.append(clairvoyant_output[1])


"""COMPUTING REWARDS WITH TRUE PROBABILITIES 100 Simulations for Greedy Algorithm"""

optimum_means = []
optimum_std_dev = []

for t in tqdm(range(T)):
    clairvoyant_output = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, rewards_parameters, real_reward_parameters=rewards_parameters, n_exp=5)
    optimum_means.append(clairvoyant_output[0])
    optimum_std_dev.append(clairvoyant_output[1])


"""COMPUTING REWARDS WITH TRUE PROBABILITIES (constant optimal reward)"""

optimum_means100 = []
attempts = []

for ex in tqdm(range(50)):
    z = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, rewards_parameters, real_reward_parameters=rewards_parameters, n_exp=5)
    attempts.append(z[0])

clairvoyant_output = max(attempts)

for t in range(T):
    optimum_means100.append(clairvoyant_output)


"""Plotting Rewards over time"""

plt.figure(figsize=(30, 10))
plt.rcParams.update({'font.size': 20})
plt.plot(time_periods, optimum_means100, color='lightgreen', linestyle='-', linewidth=3.5, label="Optimum")
plt.errorbar(time_periods, UCB_mean_rewards_per_round, yerr=UCB_std_dev_rewards_per_round, fmt='o', ecolor='lightblue', elinewidth=3, capsize=3)
plt.plot(time_periods, UCB_mean_rewards_per_round, color='blue', label='Mean Rewards')
plt.xlabel('Round')
plt.ylabel('Rewards')
plt.title('Overall Rewards When Estimating Probabilities with UCB')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(30, 10))
plt.plot(time_periods, optimum_means100, color='lightgreen', linestyle='-', linewidth=3.5, label="Optimum")
plt.errorbar(time_periods, TS_mean_rewards_per_round, yerr=TS_std_dev_rewards_per_round, fmt='o', ecolor='lightpink', elinewidth=3, capsize=3, markerfacecolor='red')
plt.plot(time_periods, TS_mean_rewards_per_round, color='red', label='Mean Rewards')
plt.xlabel('Round')
plt.ylabel('Rewards')
plt.title('Overall Rewards When Estimating Probabilities with TS')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(30, 10))
plt.plot(time_periods, optimum_means, color='#34ff00', linestyle='-', linewidth=3.5, label="Optimum")
plt.plot(time_periods, UCB_mean_rewards_per_round, color='blue', linestyle='-', label="UCB")
plt.plot(time_periods, TS_mean_rewards_per_round, color='red', linestyle='-', label="TS")
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Overall comparison of instantaneous rewards')
plt.xticks(time_periods[::30])
plt.legend()
plt.show()


"""Plotting Cumulative Regret"""

T = 365
time_periods = range(T)
diff_TS = np.cumsum([opt - ts for opt, ts in zip(optimum_means100, TS_mean_rewards_per_round)])
diff_UCB = np.cumsum([opt - ucb for opt, ucb in zip(optimum_means100, UCB_mean_rewards_per_round)])
plt.figure(figsize=(30, 10))
plt.plot(time_periods, diff_TS, color='red', linestyle='-', label="Regret TS")
plt.plot(time_periods, diff_UCB, color='blue', linestyle='-', label="Regret UCB")
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret')
plt.xticks(time_periods[::30])
plt.legend()
plt.show()
