import numpy as np
import random
import os
from collections import deque
from typing import List
from copy import copy

import pickle
import tensorflow as tf

from agents.plastic_dqn_v1 import config
from agents.base.agent import Agent
from agents.plastic_dqn_v1.aux import print_transiction, mkdir

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


class LearnBuffer:
    def __init__(self):
        self.buffer = list()
        
    def parse_episode(self, episodes_transitions: List[Transition],
                      verbose: bool = False) -> list:
        if len(episodes_transitions) == 0:
            return []
        
        # Remove last actions without ball:
        last_reward = copy(episodes_transitions[-1].reward)
        for idx in range(len(episodes_transitions) - 1, -1, -1):
            # Has ball:
            if episodes_transitions[idx].obs[5] > 0:
                episodes_transitions = episodes_transitions[:idx + 1]
                break
            # No ball:
            elif episodes_transitions[idx].obs[5] < 0:
                pass
            else:
                raise ValueError("Features has ball, wrong value!!")
        else:
            return []
        
        # selected wrong action?:
        if episodes_transitions[-1].correct_action is False and last_reward > 0:
            episodes_transitions[-1].reward = -1
        else:
            episodes_transitions[-1].reward = last_reward
        episodes_transitions[-1].done = True
        
        if verbose and random.random() > 0.99:
            print("\n ** Transictions:")
            #for el in episodes_transitions:
                #print_transiction(el.to_tuple(), self.actions, simplex=True)
            print('**')
        
        return episodes_transitions
    
    def save_episode(self, episode: List[Transition], verbose: bool = True):
        parsed_episode = self.parse_episode(episode, verbose)
        if parsed_episode:
            self.buffer += parsed_episode
    
    
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


class ExperienceBuffer:
    def __init__(self, data: np.ndarray):
        # An array with last n steps for training
        self.replay_memory = data
    
    @classmethod
    def create(cls):
        print("[Experience Buffer] Creating")
        data = np.array([])
        return cls(data)
    
    @classmethod
    def load(cls, file_path: str):
        print("[Experience Buffer] Loading")
        with open(file_path, "rb") as fp:
            data: list = pickle.load(fp)
            data_arr = np.array(data)
        return cls(data_arr)
    
    def save_to_pickle(self, dir: str, team_name: str):
        experience_file = config.REPLAY_BUFFER_FORMAT.format(
            base_dir=dir,
            team_name=team_name)
        with open(experience_file, 'wb') as f:
            pickle.dump(self.replay_memory, f)
    
    def to_array(self) -> np.ndarray:
        return self.replay_memory
    
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