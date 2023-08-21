import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.optimize import linear_sum_assignment

from Environment import Environment
from Learners import Learner, UCBLearner, TSLearner, UCBMatching, TSMatching
from Utils import simulate_episode, test_seed, greedy_algorithm, hungarian_algorithm, get_reward, clairvoyant


"""Estimating Edge Activation Probabilities"""

n_arms = 30
edge_rate=0.07
graph_structure = np.random.binomial(1, edge_rate, (n_arms, n_arms))
graph_probabilities = np.random.uniform(0.1, 0.9, (n_arms, n_arms)) * graph_structure

"""Estimating probabilities with UCB Learner..."""
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


import numpy as np

UCB_estimated_graph_probabilities = []

for index in range(len(graph_probabilities)):
    print("Estimating Arm", index)
    arm_probabilities = UCB_Generate_Probability_Estimates(graph_probabilities[index])
    arm_probabilities = np.mean(arm_probabilities, axis=0)
    UCB_estimated_graph_probabilities.append(arm_probabilities)

UCB_estimated_graph_probabilities = np.array(UCB_estimated_graph_probabilities)
UCB_estimated_graph_probabilities = np.transpose(UCB_estimated_graph_probabilities, (1, 0, 2))


for table in UCB_estimated_graph_probabilities:
    table = table*graph_structure

"""Estimating probabilities with TS Learner..."""

def TS_Generate_Probability_Estimates(p, n_arms=30, T=365, n_experiments=100):
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


import numpy as np

TS_estimated_graph_probabilities = []

for index in range(len(graph_probabilities)):
    print("Estimating Arm", index)
    arm_probabilities = TS_Generate_Probability_Estimates(graph_probabilities[index])
    arm_probabilities = np.mean(arm_probabilities, axis=0)
    TS_estimated_graph_probabilities.append(arm_probabilities)

TS_estimated_graph_probabilities = np.array(TS_estimated_graph_probabilities)
TS_estimated_graph_probabilities = np.transpose(TS_estimated_graph_probabilities, (1, 0, 2))

new_TS_estimated_graph_probabilities = []
for table in TS_estimated_graph_probabilities:
    cleaned_table = table * graph_structure
    new_TS_estimated_graph_probabilities.append(cleaned_table)

TS_estimated_graph_probabilities = new_TS_estimated_graph_probabilities


"""Estimating Matching Reward..."""

node_classes = 3
product_classes = 3
products_per_class = 3
T = 365

means = np.random.uniform(0.2, 0.8, size=(3,3))
std_dev = np.random.uniform(0.1, 0.2, size=(3, 3))
true_reward_parameters = (means, std_dev)
customer_assignments = np.random.choice([0,1,2], size=30)

"""Estimating Matching Reward with UCB..."""

p = true_reward_parameters[0]
n_experiments = 1
T = 365

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

"""Estimating Matching Reward with TS..."""

p = true_reward_parameters[0]
n_experiments = 1

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


"""Estimating overall rewards..."""

std_dev = np.full(9, 0.05)
std_dev = std_dev.reshape(3, 3)


T = 365
avg_ucb_overall_rew = []
avg_ts_overall_rew = []
std_dev_ucb_overall_rew = []
std_dev_ts_overall_rew = []

for index in range(T):
    ucb_round_score = clairvoyant(UCB_estimated_graph_probabilities[index], graph_probabilities, customer_assignments,
                                  (ucb_means[index], std_dev), true_reward_parameters, n_exp=10)
    avg_ucb_overall_rew.append(ucb_round_score[0])
    std_dev_ucb_overall_rew.append(ucb_round_score[1])

    ts_round_score = clairvoyant(TS_estimated_graph_probabilities[index], graph_probabilities, customer_assignments,
                                 (ts_means[index], std_dev), true_reward_parameters, n_exp=10)
    avg_ts_overall_rew.append(ts_round_score[0])
    std_dev_ts_overall_rew.append(ucb_round_score[1])

optimum_means = []
optimum_std_dev = []
clairvoyant_output = clairvoyant(graph_probabilities, graph_probabilities, customer_assignments, true_reward_parameters, true_reward_parameters, n_exp=1000)

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

"""Overall reward with UCB..."""

x = np.arange(T)

plt.figure(figsize=(14, 5))

for i in range(len(x)):
    plt.plot([x[i], x[i]], [avg_ucb_overall_rew[i] - std_dev_ucb_overall_rew[i], avg_ucb_overall_rew[i] + std_dev_ucb_overall_rew[i]], color='lightgrey')

plt.plot(x, avg_ucb_overall_rew, label='Overall reward with UCB', color="blue")
plt.plot(x, optimum_means, label='Optimum', color="lightgreen", linewidth=5)

plt.errorbar(x, avg_ucb_overall_rew, yerr=std_dev_ucb_overall_rew, fmt='o', markersize=2, color="lightblue", alpha=0.5)

plt.xlabel('Time')
plt.ylabel('Average Rewards')
plt.title('Overall reward with UCB')
plt.legend()

plt.tight_layout()
plt.show()


"""Overall reward with TS"""


x = np.arange(T)

plt.figure(figsize=(14, 5))

for i in range(len(x)):
    plt.plot([x[i], x[i]], [avg_ts_overall_rew[i] - std_dev_ts_overall_rew[i], avg_ts_overall_rew[i] + std_dev_ts_overall_rew[i]], color='lightgrey')

plt.plot(x, avg_ts_overall_rew, label='Overall reward with TS', color="red")
plt.plot(x, optimum_means, label='Optimum', color="lightgreen", linewidth=5)

plt.errorbar(x, avg_ts_overall_rew, yerr=std_dev_ts_overall_rew, fmt='o', markersize=2, color="lightpink", alpha=0.5)

plt.xlabel('Time')
plt.ylabel('Average Rewards')
plt.title('Overall reward with TS')
plt.legend()

plt.tight_layout()
plt.show()