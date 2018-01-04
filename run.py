from algorithms.Hill_Climbing import HillClimbing
from algorithms.Q_Learning_Table import QLearningTable
from algorithms.Policy_Gradient import PolicyGradient

import os
import logging

# 1. clear previous log files and set up logging
if os.path.exists('logFile.log'):
    os.remove('logFile.log')
logging.basicConfig(filename='logFile.log', filemode='a', level=logging.DEBUG)

# 2.
agent = PolicyGradient("CartPole-v0")
params = agent.policy_gradient()
agent.run_episode(params, show=True)



#env = gym.make('FrozenLake-v0')
