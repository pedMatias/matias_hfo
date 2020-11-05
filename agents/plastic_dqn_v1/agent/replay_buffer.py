import numpy as np
import random
from collections import deque
from typing import List

import tensorflow as tf

from agents.base.agent import Agent

REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to


class Transition:
    def __init__(self, obs: np.ndarray, act: int, reward: int,
                 new_obs: np.ndarray, done: bool, correct_action: bool):
        self.obs = obs
        self.act = act
        self.reward = reward
        self.new_obs = new_obs
        self.done = done
        # Auxiliar var:
        self.correct_action = correct_action
    
    def to_tuple(self) -> tuple:
        return tuple(
            [self.obs, self.act, self.reward, self.new_obs, self.done])
    
    
class ReplayBuffer:
    def __init__(self, memory_size: int = REPLAY_MEMORY_SIZE):
        
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    
    def store_transition(self, curr_st: np.ndarray, action_idx: int,
                         reward: int, new_st: np.ndarray, done: int):
        """ Adds step's data to a memory replay array
        (observation space, action, reward, new observation space, done) """
        transition = np.array([curr_st, action_idx, reward, new_st, done])
        self.replay_memory.append(transition)
    
    def store_episode(self, transitions: List[Transition]):
        """ Adds step's data to a memory replay array
        (observation space, action, reward, new observation space, done) """
        if len(transitions) > 0:
            for transition in transitions:
                self.replay_memory.append(transition.to_tuple())