import numpy as np
import gym

#almost_perfect_weights = [0.73193795, 1.24724398, 1.72653795, 1.5076201]

class HillClimbing:
    ''' only works with binary action space games such as CartPole-v0 - due to action selection function '''
    def __init__(self, gameName):
        self.env = gym.make(gameName)

        # initialize weight vector (4, 1 for each observation) between -1 and 1
        self.parameters = self.construct_initial_weights()
        self.reward_threshold = self.env.spec.reward_threshold
        self.trials = self.env.spec.trials

    def construct_initial_weights(self):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            return np.random.rand(*self.env.observation_space.high.shape) * 2 - 1
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            return np.random.rand(self.env.observation_space.n) * 2 - 1

    def run_episode(self, parameters, show=False):
        observation = self.env.reset()
        totalreward = 0
        t = 0
        while True:
            t += 1
            if show:
                self.env.render()
            action = 0 if np.matmul(parameters, observation) < 0 else 1
            observation, reward, done, info = self.env.step(action)
            totalreward += reward
            if done:
                if show:
                    print("Episode finished after {} timesteps".format(t))
                break
        return totalreward

    def hill_climbing(self):
        ''' start with some randomly chosen initial weights. Every episode, add some noise to the weights,
        and keep the new weights if the agent improves. '''

        noise_scaling = 0.5
        episodes_per_update = 10
        parameters = self.construct_initial_weights()
        bestreward = 0
        iteration = 0
        while iteration < 10000:
            iteration += 1
            newparams = parameters + self.construct_initial_weights() * noise_scaling
            reward = 0
            # Instead of only running one episode to measure how good a set of weights is,
            # we run it multiple times and sum up the rewards -> more accurate representation for how good a set of weights is
            for _ in range(episodes_per_update):
                run = self.run_episode(newparams)
                reward += run
            reward /= episodes_per_update

            if reward > bestreward:
                bestreward = reward
                parameters = newparams
                if reward >= self.reward_threshold:
                    average_reward = sum([self.run_episode(parameters) for _ in range(self.trials)])/self.trials
                    if average_reward > self.reward_threshold:
                        print("Solved on iteration {iteration}\nAverage reward of {reward} over {trials} trials, exceeding reward threshold of {threshold}".
                            format(iteration=iteration, reward=average_reward, trials=self.trials, threshold=self.reward_threshold))
                        break
            print("Iteration {iteration} - Reward {reward}".format(iteration=iteration, reward=reward))
        return newparams