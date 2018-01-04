'''
* Q-Learning is a table of values for every state (row) and action (column) possible in the environment
* Make updates to Q-table using the Bellman equation, which states that the expected long-term reward for
a given action is equal to the immediate reward from the current action combined with the expected reward
from the best future action taken at the following state. In this way, we reuse our own Q-table when estimating
how to update our table for future actions!
* 
'''

# TODO rewrite this to ensure it works with cartPole
# save q-table values to check if they are consistent when considered to have solved the environment

import numpy as np
import gym

class QLearningTable:
    ''' only works with binary action space games such as CartPole-v0 - due to action selection function '''

    def __init__(self, gameName):
        self.env = gym.make(gameName)

        self.reward_threshold = self.env.spec.reward_threshold
        self.trials = self.env.spec.trials

    def run_episode(self, parameters, show=False):
        observation = self.env.reset()
        totalreward = 0
        t = 0
        while True:
            t += 1
            if show:
                self.env.render()
            #action = 0 if np.matmul(parameters, observation) < 0 else 1
            action = np.argmax(parameters[observation,:])
            observation, reward, done, info = self.env.step(action)
            totalreward += reward
            if done:
                if show:
                    print("Episode finished after {} timesteps".format(t))
                print("Reward: ", totalreward)
                break
        return totalreward

    def q_learning_table(self):
         #Initialize table with all zeros
        if isinstance(self.env.observation_space, gym.spaces.Box):
            assert len(self.env.observation_space.high.shape) == 1, "Observation space is more than 1D"
            Q = np.zeros([*self.env.observation_space.high.shape, self.env.action_space.n])
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        # Set learning parameters
        lr = .8
        y = .95
        iterations = 0
        #create lists to contain total rewards and steps per episode
        #jList = []
        rList = []
        while iterations < 10000:
            iterations += 1
            s = self.env.reset()

            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            while j < 99:
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s,:] + np.random.randn(1,self.env.action_space.n)*(1.0/(iterations)))
                #Get new state and reward from environment
                s1,r,d,_ = self.env.step(a)
                #Update Q-Table with new knowledge
                Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
                rAll += r
                s = s1
                if d:
                    break
            #jList.append(j)
            rList.append(rAll)
            rolling_score = sum(rList[max(len(rList) - self.trials, 0):])/min(self.trials, len(rList))

            print("Iteration: " + str(iterations))
            print('Current score: ' + str(rAll))
            print("Average score over last 100 trials: " + str(rolling_score))
            if rolling_score >= self.reward_threshold:
                return Q
