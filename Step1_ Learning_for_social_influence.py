import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.optimize import linear_sum_assignment


from Environment import Environment
from Learners import Learner, UCBLearner, TSLearner
from Utils import simulate_episode, test_seed, greedy_algorithm, hungarian_algorithm, get_reward, clairvoyant


n_arms = 30
edge_rate = 0.07
graph_structure = np.random.binomial(1, edge_rate, (n_arms, n_arms))
graph_probabilities = np.random.uniform(0.1, 0.9, (n_arms, n_arms)) * graph_structure

node_classes = 3
product_classes = 3
products_per_class = 3
T = 365

means = np.random.uniform(25, 100, size=(3,3))
std_dev = np.random.randint(1, 30, size=(3,3))
rewards_parameters = (means, std_dev)
customer_assignments = np.random.choice([0,1,2], size=30)



"""ESTIMATING EDGE PROBABILITIES WITH UCB"""

def UCB_Generate_Probability_Estimates(p, n_arms=30, T = 365, n_experiments=100):

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
    print("Estimating Activation Probabilities for Arm", index)
    arm_probabilities = UCB_Generate_Probability_Estimates(graph_probabilities[index])
    arm_probabilities = np.mean(arm_probabilities, axis=0)
    UCB_estimated_graph_probabilities.append(arm_probabilities)

UCB_estimated_graph_probabilities = np.array(UCB_estimated_graph_probabilities)
UCB_estimated_graph_probabilities = np.transpose(UCB_estimated_graph_probabilities, (1, 0, 2))


"""ESTIMATING EDGE PROBABILITIES WITH TS"""

def TS_Generate_Probability_Estimates(p, n_arms=30, T = 365, n_experiments=100):

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
    print("Estimating Activation Probabilities for Arm", index)
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


"""Instantaneous Regrets for Edge Activation Probability Estimation"""

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

time_periods = range(1, 366)
# Plot the two lists
plt.figure(figsize=(10, 6))
plt.plot(time_periods, UCB_regret, color='blue', linestyle='-', label='UCB')
plt.plot(time_periods, TS_regret, color='red', linestyle='-', label='TS')
plt.xlabel('Time')
plt.ylabel('Instantaneous Regret')
plt.title('Instantaneous Regret of UCB and TS probability estimations')
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

for t in range(T):
    clairvoyant_output = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, rewards_parameters, real_reward_parameters=rewards_parameters, n_exp=100)
    optimum_means.append(clairvoyant_output[0])
    optimum_std_dev.append(clairvoyant_output[1])


"""COMPUTING REWARDS WITH TRUE PROBABILITIES 100.000 Simulations for Greedy Algorithm"""

optimum_means1mln = []
optimum_std_dev1mln = []
clairvoyant_output = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, rewards_parameters, real_reward_parameters=rewards_parameters, n_exp=100000)

for t in range(T):
    optimum_means1mln.append(clairvoyant_output[0])
    optimum_std_dev1mln.append(clairvoyant_output[1])


"""Plotting Rewards over time"""

plt.figure(figsize=(30, 10))
plt.plot(time_periods, optimum_means1mln, color='#34ff00', linestyle='-', linewidth=3.5, label="Optimum")
plt.plot(time_periods, UCB_mean_rewards_per_round, color='blue', linestyle='-', label="UCB")
plt.plot(time_periods, TS_mean_rewards_per_round, color='red', linestyle='-', label="TS")
plt.xlabel('Time',fontsize=20)
plt.ylabel('Reward',fontsize=20)
plt.title('Instantaneous Reward',fontsize=20)
plt.xticks(time_periods[::30])
plt.legend()
plt.show()


plt.figure(figsize=(30, 10))
plt.plot(time_periods, optimum_means, color='#34ff00', linestyle='-', linewidth=3.5, label="Optimum")
plt.plot(time_periods, UCB_mean_rewards_per_round, color='blue', linestyle='-', label="UCB")
plt.plot(time_periods, TS_mean_rewards_per_round, color='red', linestyle='-', label="TS")
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Instantaneous Reward')
plt.xticks(time_periods[::30])
plt.legend()
plt.show()