import gym
import logging

# 2. initialize the environment
env = gym.make('SpaceInvaders-v0')
print(env.action_space)
print(env.observation_space)  #.high, .low

## TODO work on visualization of observations

# 3. determine how many episodes to run
for i_episode in range(1):
    observation = env.reset()  # reset returns initial observation
    for t in range(1000):
        # 4. display the results of taking the action at each timestep
        env.render()
        # 5. determine what action to take
        action = env.action_space.sample()  # take a random action
        # 6. gather information from the environment after action is taken
        observation, reward, done, info = env.step(action)
        print((observation, reward, done, info))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
