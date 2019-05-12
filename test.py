from dual_goal_maze_env import DualGoalMaze
from tf_agents.environments import utils
from tf_agents.environments import wrappers
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

environment = DualGoalMaze()
stats_env = wrappers.RunStats(environment)

utils.validate_py_environment(stats_env, episodes=5)

time_step = stats_env.reset()
rewards = []
steps = []
num_episodes = 5

for _ in range(num_episodes):
  episode_reward = 0
  episode_steps = 0
  while not time_step.is_last():
    action = np.random.randint(0, 4)
    time_step = stats_env.step(action)
    episode_steps += 1
    episode_reward += time_step.reward
  rewards.append(episode_reward)
  steps.append(episode_steps)

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)