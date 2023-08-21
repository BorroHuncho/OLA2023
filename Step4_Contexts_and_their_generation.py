# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:28:35 2023

@author: Utente
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

class Learner:
    def __init__(self, n_arms) -> None:
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, [reward])
        self.t += 1


class UCBLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)  # count the number of times each arm has been pulled
        self.confidence = np.array([np.inf] * n_arms)

    def pull_arm(self):
        upper_confidence_bound = self.empirical_means + self.confidence
        return np.random.choice(np.where(upper_confidence_bound == upper_confidence_bound.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.n_pulls[pulled_arm] += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (
                self.n_pulls[pulled_arm] - 1) + reward) / self.n_pulls[pulled_arm]
        for a in range(self.n_arms):
            # n_samples = max(1, self.n_pulls[a])
            n_samples = self.n_pulls[a]
            self.confidence[a] = np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf

        self.update_observations(pulled_arm, reward)

    def expectations(self):
        return self.empirical_means


class TSLearner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.means = np.zeros(n_arms)  # Initialize the means array with zeros

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward
        for arm in range(self.n_arms):
            alpha = self.beta_parameters[arm, 0]
            beta = self.beta_parameters[arm, 1]
            if alpha == 1 and beta == 1:
                self.means[arm] = 0
            else:
                self.means[arm] = alpha / (alpha + beta)

    def expectations(self):
        return self.means

import numpy as np

class Environment:
    def __init__(self, probabilities):
        self.probabilities = np.array(probabilities).flatten()
        self.n_arms = len(self.probabilities)
        
    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
    
    def get_features(self):
        # Generate observed features randomly from a normal distribution
        # For this example, let's assume there are 2 features for each arm
        n_features = 2
        features = np.random.normal(0, 1, (self.n_arms, n_features))
        return features

class ContextualUCBLearner(UCBLearner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.contexts = []
        
    def update_contexts(self, contexts):
        self.contexts = contexts

class ContextualTSLearner(TSLearner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.contexts = []
        
    def update_contexts(self, contexts):
        self.contexts = contexts

class ContextGenerator:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, features):
        self.kmeans.fit(features)

    def predict(self, features):
        return self.kmeans.predict(features)

# Define the number of clusters for context generation
n_clusters = 5

# Create an instance of the ContextGenerator class
context_generator = ContextGenerator(n_clusters=n_clusters)

# Define the number of arms
n_arms = 30

# Define the edge rate
edge_rate = 0.07

# Define the time horizon
T = 365

# Create an instance of the Environment class
env = Environment(probabilities=np.random.rand(n_arms))

# Create instances of the ContextualUCBLearner and ContextualTSLearner classes
ucb_learner = ContextualUCBLearner(n_arms=n_arms)
ts_learner = ContextualTSLearner(n_arms=n_arms)

# Define the time interval for context generation
context_interval = 14

# Simulate the contextual bandit problem
for t in range(1, T+1):
    # Get the features associated with the arms
    features = env.get_features()

    # Update the contexts every two weeks
    if t % context_interval == 0:
        context_generator.fit(features)
        contexts = context_generator.predict(features)
        ucb_learner.update_contexts(contexts)
        ts_learner.update_contexts(contexts)

    # Pull the arms and update the learners
    pulled_arm_ucb = ucb_learner.pull_arm()
    reward_ucb = env.round(pulled_arm_ucb)
    ucb_learner.update(pulled_arm_ucb, reward_ucb)

    pulled_arm_ts = ts_learner.pull_arm()
    reward_ts = env.round(pulled_arm_ts)
    ts_learner.update(pulled_arm_ts, reward_ts)

#SECOND PART OF THE CODE



# Define the optimal arm
optimal_arm = np.argmax(env.probabilities)

# Define the number of runs for averaging
n_runs = 100

# Define arrays to store the performance metrics
cumulative_regret_ucb = np.zeros(T)
cumulative_reward_ucb = np.zeros(T)
instantaneous_regret_ucb = np.zeros(T)
instantaneous_reward_ucb = np.zeros(T)

cumulative_regret_ts = np.zeros(T)
cumulative_reward_ts = np.zeros(T)
instantaneous_regret_ts = np.zeros(T)
instantaneous_reward_ts = np.zeros(T)


# Define arrays to store the performance metrics for the clairvoyant
cumulative_reward_clairvoyant = np.zeros(T)
instantaneous_reward_clairvoyant = np.zeros(T)

# Simulate the contextual bandit problem for multiple runs
for run in range(n_runs):
    ucb_learner = ContextualUCBLearner(n_arms=n_arms)
    ts_learner = ContextualTSLearner(n_arms=n_arms)
    context_generator = ContextGenerator(n_clusters=n_clusters)
    for t in range(1, T+1):
        features = env.get_features()
        if t % context_interval == 0:
            context_generator.fit(features)
            contexts = context_generator.predict(features)
            ucb_learner.update_contexts(contexts)
            ts_learner.update_contexts(contexts)

        pulled_arm_ucb = ucb_learner.pull_arm()
        reward_ucb = env.round(pulled_arm_ucb)
        ucb_learner.update(pulled_arm_ucb, reward_ucb)

        pulled_arm_ts = ts_learner.pull_arm()
        reward_ts = env.round(pulled_arm_ts)
        ts_learner.update(pulled_arm_ts, reward_ts)

        optimal_reward = env.round(optimal_arm)

        cumulative_regret_ucb[t-1] += optimal_reward - reward_ucb
        cumulative_reward_ucb[t-1] += reward_ucb
        instantaneous_regret_ucb[t-1] += optimal_reward - reward_ucb
        instantaneous_reward_ucb[t-1] += reward_ucb

        cumulative_regret_ts[t-1] += optimal_reward - reward_ts
        cumulative_reward_ts[t-1] += reward_ts
        instantaneous_regret_ts[t-1] += optimal_reward - reward_ts
        instantaneous_reward_ts[t-1] += reward_ts

        cumulative_reward_clairvoyant[t-1] += optimal_reward
        instantaneous_reward_clairvoyant[t-1] += optimal_reward

# Calculate the average and standard deviation of the performance metrics
cumulative_regret_ucb /= n_runs
cumulative_reward_ucb /= n_runs
instantaneous_regret_ucb /= n_runs
instantaneous_reward_ucb /= n_runs

cumulative_regret_ts /= n_runs
cumulative_reward_ts /= n_runs
instantaneous_regret_ts /= n_runs
instantaneous_reward_ts /= n_runs

cumulative_reward_clairvoyant = n_runs
instantaneous_reward_clairvoyant /= n_runs

# Calculate standard deviation for instantaneous rewards
std_instantaneous_reward_ucb = np.std(instantaneous_reward_ucb)
std_instantaneous_reward_ts = np.std(instantaneous_reward_ts)

# Calculate the cumulative reward for the clairvoyant
cumulative_reward_clairvoyant = np.cumsum(instantaneous_reward_clairvoyant)

# Set the instantaneous reward for the clairvoyant to a constant value of 1
instantaneous_reward_clairvoyant = np.ones(T)

# Calculate the cumulative rewards for UCB, TS, and clairvoyant
cumulative_reward_ucb = np.cumsum(instantaneous_reward_ucb)
cumulative_reward_ts = np.cumsum(instantaneous_reward_ts)
cumulative_reward_clairvoyant = np.cumsum(instantaneous_reward_clairvoyant)

# Plot the performance metrics
plt.figure(figsize=(12, 8))
plt.plot(instantaneous_reward_ucb, label='UCB Instantaneous Reward', color='blue')
plt.plot(instantaneous_reward_ts, label='TS Instantaneous Reward', color='red')
plt.plot(instantaneous_reward_clairvoyant, label='Clairvoyant Instantaneous Reward', color='green')
plt.fill_between(range(T), instantaneous_reward_ucb - std_instantaneous_reward_ucb, instantaneous_reward_ucb + std_instantaneous_reward_ucb, alpha=0.2, color='blue')
plt.fill_between(range(T), instantaneous_reward_ts - std_instantaneous_reward_ts, instantaneous_reward_ts + std_instantaneous_reward_ts, alpha=0.2, color='red')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.legend()
plt.title('Instantaneous Rewards + Standard Deviation')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(instantaneous_reward_ucb, label='UCB Instantaneous Reward', color='blue')
plt.plot(instantaneous_reward_ts, label='TS Instantaneous Reward', color='red')
plt.plot(instantaneous_reward_clairvoyant, label='Clairvoyant Instantaneous Reward', color='green')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.legend()
plt.title('Instantaneous Reward')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(cumulative_regret_ucb, label='UCB Cumulative Regret', color='blue')
plt.plot(cumulative_regret_ts, label='TS Cumulative Regret', color='red')
plt.xlabel('Time')
plt.ylabel('Regret')
plt.legend()
plt.title('Cumulative Regrets')
plt.show()

# Plot the cumulative rewards
plt.figure(figsize=(12, 8))
plt.plot(cumulative_reward_ucb, label='UCB Cumulative Reward', color='blue')
plt.plot(cumulative_reward_ts, label='TS Cumulative Reward', color='red')
plt.plot(cumulative_reward_clairvoyant, label='Clairvoyant Cumulative Reward', color='green')
plt.xlabel('Time')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.title('Cumulative Rewards')
plt.show()
