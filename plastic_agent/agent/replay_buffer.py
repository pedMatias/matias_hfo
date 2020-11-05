import numpy as np
import os
import random
from copy import copy
from typing import List

import pickle

from agents.plastic_dqn_v1 import config

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
    """ Buffer used during game. Saves and decides which steps to save """
    def __init__(self, buffer: list = None):
        if buffer is None:
            buffer = list()
        self.buffer = buffer
    
    @classmethod
    def load_from_file(cls, file_name: str):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, LearnBuffer):
            return data
        elif isinstance(data, list):
            return cls(data)
        else:
            raise ValueError(f"Unexpected type of {file_name}")
        
    def to_list(self):
        return self.buffer
        
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
    
    def export_to_pickle(self, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(self.buffer, f)


class ExperienceBuffer:
    """ Contains all the experience the agent retain during training """
    def __init__(self, data: list):
        self._data = data
    
    @classmethod
    def create_by_merge_files(cls, directory: str, team_name: str):
        print("[Experience Buffer] Merging smaller files")
        step = 0
        exp_episodes = list()
        while True:
            # Check if file exists:
            experience_file = config.EXPERIENCE_BUFFER_FORMAT.format(step=step)
            data_file = os.path.join(directory, team_name, experience_file)
            if not os.path.isfile(data_file):
                break
            # Load Learn Buffer:
            learn_buffer = LearnBuffer.load_from_file(data_file)
            exp_episodes += learn_buffer.to_list()
            # Inc step:
            print(f"Add stage {step} data. SIZE={len(learn_buffer.to_list())}")
            step += 1
    
        experience_buffer = cls(exp_episodes)
        experience_buffer.save_to_pickle(dir=directory, team_name=team_name)
        return experience_buffer
    
    @classmethod
    def load(cls, file_path: str):
        print("[Experience Buffer] Loading")
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return cls(data)
    
    def to_list(self) -> list:
        return self._data
    
    def to_array(self) -> np.ndarray:
        return np.array(self._data)
    
    def save_to_pickle(self, dir: str, team_name: str):
        experience_file = config.REPLAY_BUFFER_FORMAT.format(
            base_dir=dir,
            team_name=team_name)
        with open(experience_file, 'wb') as f:
            pickle.dump(self.to_array(), f)
    
    def store_transition(self, curr_st: np.ndarray, action_idx: int,
                         reward: int, new_st: np.ndarray, done: int):
        """ Adds step's data to a memory replay array
        (observation space, action, reward, new observation space, done) """
        transition = np.array([curr_st, action_idx, reward, new_st, done])
        self._data.append(transition)
    
    def store_episode(self, transitions: List[Transition]):
        """ Adds step's data to a memory replay array
        (observation space, action, reward, new observation space, done) """
        if len(transitions) > 0:
            for transition in transitions:
                self._data.append(transition.to_tuple())
