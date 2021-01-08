import json
import numpy as np
import os
from copy import copy
from typing import List

import pickle

from plastic_policy_agent import config


class Transition:
    def __init__(self, obs: np.ndarray, act: int, reward: int,
                 new_obs: np.ndarray, done: bool):
        self.obs = obs
        self.act = act
        self.reward = reward
        self.new_obs = new_obs
        self.done = done
    
    def to_tuple(self) -> tuple:
        return tuple(
            [self.obs, self.act, self.reward, self.new_obs, self.done])


class LearnBuffer:
    """ Buffer used during game. Saves and decides which steps to save """
    def __init__(self, dqn_buffer: list = None,
                 team_experience_buffer: list = None):
        # Buffer used to train DQN
        if dqn_buffer:
            self.dqn_buffer = dqn_buffer
        else:
            self.dqn_buffer = list()
        # Team experience Buffer. Used to train team model:
        if team_experience_buffer:
            self.team_experience_buffer = team_experience_buffer
        else:
            self.team_experience_buffer = list()
    
    @classmethod
    def load_team_experience_from_file(cls, file_name: str):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, LearnBuffer):
            return data
        elif isinstance(data, list):
            return cls(team_experience_buffer=data)
        else:
            raise ValueError(f"Unexpected type of {file_name}")
        
    def parse_episode(self, episodes_transitions: List[Transition]) -> list:
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
        
        episodes_transitions[-1].reward = last_reward
        episodes_transitions[-1].done = True
        return episodes_transitions
    
    def save_episode(self, episode: List[Transition]):
        self.team_experience_buffer += episode
        # DQN Buffer:
        parsed_episode = self.parse_episode(episode.copy())
        if parsed_episode:
            self.dqn_buffer += parsed_episode
    
    def export_dqn_buffer_to_pickle(self, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(self.dqn_buffer, f)
    
    def export_team_experience_to_pickle(self, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(self.team_experience_buffer, f)


class ExperienceBuffer:
    """ Contains all the experience the agent retain during training """
    def __init__(self, data: list):
        self._data = data
    
    @classmethod
    def create_by_merge_files(cls, directory: str):
        print("[Experience Buffer] Merging smaller files")
        step = 0
        exp_episodes = list()
        while True:
            # Check if file exists:
            experience_file = config.TEAM_EXPERIENCE_BUFFER_FORMAT.format(step=step)
            data_file = os.path.join(directory, experience_file)
            if not os.path.isfile(data_file):
                print(f"CANT find {data_file}")
                break
            # Load Learn Buffer:
            learn_buffer = LearnBuffer.load_team_experience_from_file(data_file)
            exp_episodes += learn_buffer.team_experience_buffer
            # Inc step:
            print(f"Add stage {step} data. "
                  f"SIZE={len(learn_buffer.team_experience_buffer)}")
            step += 1
    
        experience_buffer = cls(exp_episodes)
        experience_buffer.save_to_pickle(dir=directory)
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
    
    def save_to_pickle(self, dir: str):
        experience_file = os.path.join(dir, config.REPLAY_BUFFER_FORMAT)
        with open(experience_file, 'wb') as f:
            pickle.dump(self.to_array(), f)
        # Write metrics:
        experience_file_metrics = experience_file + ".metrics.txt"
        metrics = dict(number_of_steps=len(self.to_list()))
        with open(experience_file_metrics, 'w+') as fp:
            json.dump(metrics, fp)
    
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
