import numpy as np
import matplotlib.pyplot as plt
import warnings

class N_Armed_Bandit():
    def __init__(self, n_arms, eps, iterations):
        self.n_arms = n_arms
        self.arms = np.random.rand(n_arms)
        self.eps = eps
        self.iterations = iterations

        self.action_values = np.empty((self.n_arms, self.iterations)) * np.NAN

        for iteration in range(self.iterations):
            print('iteration:', iteration)
            move = self.policy()
            print('move:', move)
            outcome = self.reward(self.arms[move])
            self.action_values[move, iteration] = outcome

        print('original', self.arms)

        scores_history = np.nansum(self.action_values, axis=0)

        cumulative_scores = np.cumsum(scores_history)
        xs = np.arange(1, iterations + 1, 1)

        plt.scatter(xs, cumulative_scores / xs)
        plt.show()

    def reward(self, prob):
        reward = 0
        for i in range(10):
            if np.random.random() < prob:
                reward += 1
        return reward

    def policy(self):
        strategy_choice = np.random.random()
        if strategy_choice < (1-self.eps):
            print("greedy")
            means = np.nanmean(self.action_values, axis=1)
            print('means:', means)
            return np.argmax(means)
        else:
            print('random')
            return np.random.randint(0, self.n_arms)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    kwargs = dict(n_arms=10, eps=0.1, iterations=1000)
    N_Armed_Bandit(**kwargs)
