import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.optimize import linear_sum_assignment


from Environment import Environment, NonStationaryEnvironment
from Learners import Learner, UCBLearner, TSLearner, CUSUM, CUSUMUCB, SW_UCBLearner
from Utils import simulate_episode, test_seed, greedy_algorithm, hungarian_algorithm, get_reward, clairvoyant


"""Classes of product and Classes of customers"""

node_classes = 3
product_classes = 3
products_per_class = 3

means = np.random.uniform(10, 20, (3,3))
std_dev = np.ones((3,3))
rewards_parameters = (means, std_dev)

customer_assignments = np.random.choice([0,1,2], size=30)

n_arms = 30
n_phases = 3
T = 365
window_size = int(T**0.5)
n_experiments = 50


def generate_graph_probabilities(n_nodes, edge_rate):
    graph_structure = np.random.binomial(1, edge_rate, (n_nodes, n_nodes))
    graph_probabilities = np.random.uniform(0.1, 0.9, (n_nodes, n_nodes)) * graph_structure
    return graph_probabilities

n_nodes = 30
edge_rate = 0.05

prob_phase1 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))
prob_phase2 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))
prob_phase3 = np.reshape(generate_graph_probabilities(n_nodes=n_nodes,edge_rate=edge_rate),(30,30))

# Array containing three (30*30) different probabilities tables.
p = np.stack((prob_phase1, prob_phase2, prob_phase3), axis=0)


# Array K will contain 30 arrays containing each 3 rows: row[i] of probability table of phase1, row[i] of the one of phase2, row[i] of the one of phase3.
K = np.array([p[:, i] for i in range(p.shape[1])])



"""Probability estimators"""
def SW_Generate_Probability_Estimates(p, n_arms=n_arms, n_phases=n_phases, T=T, window_size=window_size, n_experiments=n_experiments):
    phases_len = int(T / n_phases)
    swucb_rewards_per_experiment = []

    experimentS_means_at_each_round = np.empty((n_experiments, T, n_arms))

    for e in tqdm(range(0, n_experiments)):
        # Initialize environment
        swucb_env = NonStationaryEnvironment(probabilities=p, horizon=T)
        # Initialize learner
        swucb_learner = SW_UCBLearner(n_arms=n_arms, window_size=window_size)

        for t in range(0, T):
            # SW-UCB Learner
            pulled_arm = swucb_learner.pull_arm()
            reward = swucb_env.round(pulled_arm)
            swucb_learner.update(pulled_arm, reward)

            # At each round memorize a copy of the means of each arm
            expected_rew = swucb_learner.expectations()
            experimentS_means_at_each_round[e, t] = expected_rew.copy()

    return experimentS_means_at_each_round


def CUSUM_Generate_Probability_Estimates(p, n_arms=n_arms, n_phases=n_phases, T=T, n_experiments=n_experiments):
    phases_len = int(T / n_phases)
    cusum_rewards_per_experiment = []

    experimentS_means_at_each_round = np.empty((n_experiments, T, n_arms))

    for e in tqdm(range(0, n_experiments)):
        # Initialize environment
        cusum_env = NonStationaryEnvironment(probabilities=p, horizon=T)
        # Initialize learner
        cusum_learner = CUSUMUCB(n_arms=n_arms)

        for t in range(0, T):
            # SW-UCB Learner
            pulled_arm = cusum_learner.pull_arm()
            reward = cusum_env.round(pulled_arm)
            cusum_learner.update(pulled_arm, reward)

            # At each round memorize a copy of the means of each arm
            expected_rew = cusum_learner.expectations()
            experimentS_means_at_each_round[e, t] = expected_rew.copy()

    return experimentS_means_at_each_round


"""Estimating non-stationary activation probabilities with Sliding Window UCB"""

SW_rounds_probabilities_for_each_arm = []

for index in range(len(K)):
    print("Learning for row:",index)
    estimates = SW_Generate_Probability_Estimates(K[index])
    SW_rounds_probabilities_for_each_arm.append(estimates)

SW_rounds_probabilities_for_each_arm = np.mean(SW_rounds_probabilities_for_each_arm, axis=1)


"""Estimating non-stationary activation probabilities with CUSUM UCB"""

CUSUM_rounds_probabilities_for_each_arm = []

for index in range(len(K)):
    print("Learning for row:",index)
    estimates = CUSUM_Generate_Probability_Estimates(K[index])
    CUSUM_rounds_probabilities_for_each_arm.append(estimates)

CUSUM_rounds_probabilities_for_each_arm = np.mean(CUSUM_rounds_probabilities_for_each_arm, axis=1)


"""Adjusting output so to have one graph probability table for each round, for both CUSUM and SW UCB"""
def Reshape(LIST):
  # Convert the lists into a NumPy array
  array_of_lists = np.array(LIST)
  # Transpose the array to swap the axes
  transposed_array = array_of_lists.T
  # Split the transposed array into separate arrays along axis=1
  return np.split(transposed_array, transposed_array.shape[1], axis=1)

estimated_tables_SW = Reshape(SW_rounds_probabilities_for_each_arm)
estimated_tables_CUSUM = Reshape(CUSUM_rounds_probabilities_for_each_arm)


"""Repeating for each round the corresponding true probability table, depending on the phase"""

phases_array = np.empty((T, n_nodes, n_nodes))
T = 365
phases_len = int(T / n_phases)

for t in range(T):
    if t <= 121:
        phases_array[t] = p[0]
    if t in range(121, 243):
        phases_array[t] = p[1]
    if t >243:
        phases_array[t] = p[2]


"""Regret in Probability Estimation"""

sw_diff = []
cusum_diff = []
opt = []

cumulative_sw_regret = 0
cumulative_cusum_regret = 0

for index in range(len(phases_array)):
    sw_difference = np.abs(phases_array[index] - estimated_tables_SW[index])
    sw_tot_difference = np.sum(sw_difference)
    cumulative_sw_regret += sw_tot_difference
    sw_diff.append(cumulative_sw_regret)

    cusum_difference = np.abs(phases_array[index] - estimated_tables_CUSUM[index])
    cusum_tot_difference = np.sum(cusum_difference)
    cumulative_cusum_regret += cusum_tot_difference
    cusum_diff.append(cumulative_cusum_regret)

plt.plot(sw_diff, color='red', label='Cumulative SW Regret')
plt.plot(cusum_diff, color='blue', label='Cumulative CUSUM Regret')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret for Activation Probability Estimation')

# Add a legend
plt.legend()

# Show the plot
plt.show()

"""Computing overall rewards with SW UCB estimated probabilities"""

n_exp = 10

SW_mean_rewards_per_round = []
SW_std_dev_rewards_per_round = []
for table in tqdm(range(len(estimated_tables_SW))):
    true_p = phases_array[table]
    table = np.reshape(estimated_tables_SW[table],(30,30))
    clairvoyant_output = clairvoyant(table, true_p, customer_assignments, rewards_parameters, rewards_parameters, n_exp)
    SW_mean_rewards_per_round.append(clairvoyant_output[0])
    SW_std_dev_rewards_per_round.append(clairvoyant_output[1])


"""Computing overall rewards with CUSUM UCB estimated probabilities"""

CUSUM_mean_rewards_per_round = []
CUSUM_std_dev_rewards_per_round = []
for table in range(len(estimated_tables_CUSUM)):
    true_p = phases_array[table]
    table = np.reshape(estimated_tables_CUSUM[table],(30,30))
    clairvoyant_output = clairvoyant(table, true_p, customer_assignments, rewards_parameters, rewards_parameters, n_exp)
    CUSUM_mean_rewards_per_round.append(clairvoyant_output[0])
    CUSUM_std_dev_rewards_per_round.append(clairvoyant_output[1])

"""Computing optimum"""

optimum_means = []
optimum_std_dev = []
for table in p:
    clairvoyant_output = clairvoyant(table, table, customer_assignments, rewards_parameters, rewards_parameters, n_exp=1000000)
    for i in range(int(T / n_phases)+1):
        optimum_means.append(clairvoyant_output[0])
        optimum_std_dev.append(clairvoyant_output[1])

optimum_means = optimum_means[:-1]
optimum_std_dev = optimum_std_dev[:-1]


"""Plotting Instantaneous Reward for the SW UCB case"""

plt.figure(figsize=(10, 6))

time_periods = range(len(SW_mean_rewards_per_round))

for t in time_periods:
    mean = SW_mean_rewards_per_round[t]
    std_dev = SW_std_dev_rewards_per_round[t]
    plt.vlines(t, mean - std_dev, mean + std_dev, color='lightgrey')

plt.plot(time_periods, optimum_means, color='green', linestyle='-', linewidth=5)
plt.plot(time_periods, SW_mean_rewards_per_round, color='red', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Sliding Window UCB')
plt.xticks(time_periods[::30])
plt.figure(figsize=(10, 6))
plt.show()


"""Plotting Instantaneous Reward for the CUSUM UCB case"""

plt.figure(figsize=(12, 7))
time_periods = range(len(CUSUM_mean_rewards_per_round))

for t in time_periods:
    mean = CUSUM_mean_rewards_per_round[t]
    std_dev = CUSUM_std_dev_rewards_per_round[t]
    plt.vlines(t, mean - std_dev, mean + std_dev, color='lightgrey')
plt.plot(time_periods, optimum_means, color='green', linestyle='-', linewidth=5)
plt.plot(time_periods, CUSUM_mean_rewards_per_round, color='blue', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Change Detection CUSUM UCB')
plt.xticks(time_periods[::30])
plt.figure(figsize=(10, 6))
plt.show()




"""Comparison of instantaneous rewards"""

time_periods = range(len(SW_mean_rewards_per_round))
plt.plot(time_periods, SW_mean_rewards_per_round, color='red', linestyle='-', label = "SW_UCB")
plt.plot(time_periods, CUSUM_mean_rewards_per_round, color="blue",linestyle="-", label = "CD_UCB")
plt.xlabel('Time')
plt.ylabel('Mean Reward Per Round')
plt.title('Istantaneous Rewards')
plt.xticks(time_periods[::30])
plt.figure(figsize=(10, 6))
plt.show()


"""Comparison of instantaneous regrets"""

time_periods = range(len(SW_mean_rewards_per_round))
plt.plot(time_periods, [x - y for x, y in zip(optimum_means, SW_mean_rewards_per_round)], color='red', linestyle='-', label = "SW_UCB")
plt.plot(time_periods, [x - y for x, y in zip(optimum_means, CUSUM_mean_rewards_per_round)], color="blue",linestyle="-", label = "CD_UCB")
plt.xlabel('Time')
plt.ylabel('Regret per Round')
plt.title('Istantaneous Regrets')
plt.xticks(time_periods[::30])
plt.figure(figsize=(10, 6))
plt.show()



"""Comparison of overall cumulative regrets"""

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
SW_cumulative_regret = [sum([abs(x - y) for x, y in zip(optimum_means[:t+1], SW_mean_rewards_per_round[:t+1])]) for t in time_periods]
CD_cumulative_regret = [sum([abs(x - y) for x, y in zip(optimum_means[:t+1], CUSUM_mean_rewards_per_round[:t+1])]) for t in time_periods]
plt.plot(time_periods, SW_cumulative_regret, color='red', linestyle='-', label="SW_UCB")
plt.plot(time_periods, CD_cumulative_regret, color="blue", linestyle="-", label="CD_UCB")
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regrets')
plt.xticks(time_periods[::30])
plt.legend()
plt.figure(figsize=(10, 6))
plt.show()