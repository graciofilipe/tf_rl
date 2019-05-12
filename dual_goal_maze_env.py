import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

class DualGoalMaze(py_environment.PyEnvironment):

    def __init__(self):

        self._action_spec = array_spec.BoundedArraySpec(
            shape=np.array([1]), dtype=np.int32, minimum=0, maximum=3, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.int32,
            minimum=0,
            maximum=5,
            name='observation')

        self._state = np.array([0, 0, 0], dtype=np.int32)
        self._end_state = np.array([0, 0, 1], dtype=np.int32)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_state(self):
        return self._state

    def get_reward(self):
        if (self._state == self._end_state).all():
            return 1
        else:
            return -1

    def _reset(self):
        self._state = np.array([0, 0, 0], dtype=np.int32)
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.take_action(action)
        reward = self.get_reward()
        self._episode_ended = (self._state == self._end_state).all()

        if self._episode_ended:
            return ts.termination(self.get_state(), reward=reward)
        else:
            return ts.transition(self.get_state(), reward=reward, discount=0.8)


    def take_action(self, action):
        if action == 0: # up
            self._state[1] = np.min([self._state[1] + 1, 5])
        if action == 1: # down
            self._state[1] = np.max([self._state[1] - 1, 0])
        if action == 2: # left
            self._state[0] = np.max([self._state[0] - 1, 0])
        if action == 3: # right
            self._state[0] = np.min([self._state[0] + 1, 5])

        if (self._state == np.array([5, 5, 0])).all():
            self._state = np.array([5, 5, 1], dtype=np.int32)