import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.optimize import linear_sum_assignment


from Environment import Environment
from Learners import Learner, UCBLearner, TSLearner, UCBMatching, TSMatching
from Utils import simulate_episode, test_seed, greedy_algorithm, hungarian_algorithm, get_reward, clairvoyant

n_arms = 30
edge_rate=0.07
graph_structure = np.random.binomial(1, edge_rate, (n_arms, n_arms))
graph_probabilities = np.random.uniform(0.1, 0.9, (n_arms, n_arms)) * graph_structure



node_classes = 3
product_classes = 3
products_per_class = 3
T = 365

means = np.random.uniform(0.2, 0.8, size=(3,3))
std_dev = np.random.uniform(0.1, 0.2, size=(3, 3))
true_reward_parameters = (means, std_dev)
customer_assignments = np.random.choice([0,1,2], size=30)



"""Estimating means with MatchingUCB"""

p = true_reward_parameters[0]
n_experiments = 3000

learner = UCBMatching(p.size, *p.shape)
rewards_per_experiment = []
means_per_experiment = []
env = Environment(p)

means = []
for exp in tqdm(range(n_experiments)):
    experiment_means = []

    for t in range(T):
        pulled_arms = learner.pull_arm()
        reward = env.round(pulled_arms)
        learner.update(pulled_arms, reward)
        x = learner.expectations()
        experiment_means.append(np.array(x).reshape(3, 3))

    means.append(experiment_means)

means = np.array(means)
ucb_means = np.mean(means, axis=0)

"""Estimating means with MatchingTS"""

p = true_reward_parameters[0]
n_experiments = 3000

learner = TSMatching(p.size, *p.shape)
rewards_per_experiment = []
means_per_experiment = []
env = Environment(p)

means = []
for exp in tqdm(range(n_experiments)):
    experiment_means = []

    for t in range(T):
        pulled_arms = learner.pull_arm()
        reward = env.round(pulled_arms)
        learner.update(pulled_arms, reward)
        x = learner.expectations()
        experiment_means.append(np.array(x).reshape(3, 3))

    means.append(experiment_means)

means = np.array(means)
ts_means = np.mean(means, axis=0)



"""Computing Regrets for Matching Problem"""

row_ind, col_ind = linear_sum_assignment(-p)
optimum = p[row_ind, col_ind].sum()

ucb_matching_rewards = []
ucb_matching_regret = []

for i in range(len(ucb_means)):
    matrix = ucb_means[i]
    row_ind, col_ind = linear_sum_assignment(-matrix)
    matching_rewards_sum = matrix[row_ind, col_ind].sum()
    ucb_matching_rewards.append(matching_rewards_sum)
    ucb_matching_regret.append(optimum - matching_rewards_sum)

ts_matching_rewards = []
ts_matching_regret = []

for i in range(len(ts_means)):
    matrix = ts_means[i]
    row_ind, col_ind = linear_sum_assignment(-matrix)
    matching_rewards_sum = matrix[row_ind, col_ind].sum()
    ts_matching_rewards.append(matching_rewards_sum)
    ts_matching_regret.append(optimum - matching_rewards_sum)

optimums = []
for i in range(len(ucb_means)):
    optimums.append(optimum)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(ts_matching_rewards, marker='o', markersize=1,label='TS Matching Rewards', color="red")
plt.plot(optimums, marker='o', markersize=1,label='Optimum Rewards', color="green")
plt.xlabel('Iteration')
plt.ylabel('Rewards')
plt.title('TS Matching Rewards')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ts_matching_regret, marker='o',markersize=1, color='red', label='TS Matching Regret')
plt.xlabel('Iteration')
plt.ylabel('Regret')
plt.title('TS Matching Regret')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(ucb_matching_rewards, marker='o', markersize=1,label='UCB Matching Rewards', color = "blue")
plt.plot(optimums, marker='o', markersize=1,label='Optimum Rewards', color="green")
plt.xlabel('Iteration')
plt.ylabel('Rewards')
plt.title('UCB Matching Rewards')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ucb_matching_regret, marker='o', markersize=1, color='blue', label='UCB Matching Regret')
plt.xlabel('Iteration')
plt.ylabel('Regret')
plt.title('UCB Matching Regret')
plt.legend()

plt.tight_layout()
plt.show()

"""Computing overall rewards"""
"""Overall outcome when estimating matching rewards with UCBMatching"""

opt_seeds = greedy_algorithm(graph_probabilities, 3, 10000, 10)
std_dev = np.full(9, 0.05)
std_dev = std_dev.reshape(3,3)


avg_ucb_overall_rew = []
avg_ts_overall_rew = []
std_dev_ucb_overall_rew = []
std_dev_ts_overall_rew = []

for index in range(T):
    ucb_round_score = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, (ucb_means[index], std_dev), real_reward_parameters=true_reward_parameters, n_exp=100, seeds=opt_seeds)
    avg_ucb_overall_rew.append(ucb_round_score[0])
    std_dev_ucb_overall_rew.append(ucb_round_score[1])

    ts_round_score = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, (ts_means[index], std_dev), real_reward_parameters=true_reward_parameters, n_exp=100, seeds=opt_seeds)
    avg_ts_overall_rew.append(ts_round_score[0])
    std_dev_ts_overall_rew.append(ucb_round_score[1])


optimum_means = []
optimum_std_dev = []
clairvoyant_output = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, rewards_parameters=true_reward_parameters, real_reward_parameters=true_reward_parameters, n_exp=100, seeds=opt_seeds)

for t in range(T):
    optimum_means.append(clairvoyant_output[0])
    optimum_std_dev.append(clairvoyant_output[1])



"""Plotting Overall Rewards and Regret"""

opt_seeds = greedy_algorithm(graph_probabilities, 3, 100, 10)
optimum_means = []
optimum_std_dev = []
clairvoyant_output = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, true_reward_parameters, real_reward_parameters=true_reward_parameters, n_exp=100, seeds=opt_seeds)


for t in range(T):
    optimum_means.append(clairvoyant_output[0])
    optimum_std_dev.append(clairvoyant_output[1])

x = np.arange(T)
plt.figure(figsize=(14, 5))
plt.plot(x, avg_ucb_overall_rew, label='Overall reward with UCB', color="blue")
plt.plot(x, avg_ts_overall_rew, label='Overall reward with TS', color="red")
plt.plot(x, optimum_means, label='Optimum', color="lightgreen", linewidth=5)
plt.xlabel('Time')
plt.ylabel('Average Rewards')
plt.title('Comparison of overall rewards')
plt.legend()
plt.tight_layout()
plt.show()



ucb_regret = np.abs(np.subtract(optimum_means, avg_ucb_overall_rew))
ts_regret = np.abs(np.subtract(optimum_means, avg_ts_overall_rew))
plt.figure(figsize=(14, 5))
plt.plot(x, ucb_regret, label='Instantaneous Regret with UCB', color="blue")
plt.plot(x, ts_regret, label='Instantaneous Regret with TS', color="red")
plt.xlabel('Time')
plt.ylabel('Regret')
plt.title('Comparison of overall regret')
plt.legend()
plt.tight_layout()
plt.show()




ucb_regret = np.subtract(optimum_means, avg_ucb_overall_rew)
ts_regret = np.subtract(optimum_means, avg_ts_overall_rew)
cumulative_ucb_regret = np.cumsum(ucb_regret)
cumulative_ts_regret = np.cumsum(ts_regret)
plt.figure(figsize=(14, 5))
plt.plot(x, cumulative_ucb_regret, label='Cumulative Regret with UCB', color="blue")
plt.plot(x, cumulative_ts_regret, label='Cumulative Regret with TS', color="red")
plt.xlabel('Time')
plt.ylabel('Regret')
plt.title('Comparison of overall regret')
plt.legend()
plt.tight_layout()
plt.show()