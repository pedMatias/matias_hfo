import os
import pickle
import random

import numpy as np
from sklearn.neighbors import KDTree

from multi_agents import config
from multi_agents.dqn_agent.replay_buffer import Transition, ExperienceBuffer


class TeamModel:
    """ Saves an KDTree with all the initial states of any transition, and
    also a list with all the Transitions """
    
    def __init__(self, states: np.ndarray, next_states: np.ndarray):

        # KDTree:
        print("[Team Model: KDTree] Creating KDTree")
        self.model: KDTree = KDTree(states)
        
        # Matrix saving all the next states of each state. Each line i
        # corresponds to the next state of state i:
        print("[Team Model: next states] Creating Next States array")
        self.next_states: np.ndarray = np.array(next_states)
    
    @staticmethod
    def _load_experience_data(directory: str):
        rep_buffer_file = os.path.join(directory, config.REPLAY_BUFFER_FORMAT)
        if not os.path.isfile(rep_buffer_file):
            exp_buffer = ExperienceBuffer.create_by_merge_files(directory)
        else:
            exp_buffer = ExperienceBuffer.load(rep_buffer_file)
        print("File ", rep_buffer_file)
        print("Replay Buffer: ", len(exp_buffer.to_list()),
              exp_buffer.to_list()[0])
        return exp_buffer.to_array()
    
    @staticmethod
    def get_states_and_next_states(data: np.ndarray) -> (list, list):
        states = []
        next_states = []
        for transiction in data:
            states.append(transiction.obs)
            next_states.append(transiction.new_obs)
        return states, next_states
    
    @classmethod
    def create_and_save(cls, directory: str):
        # Load previous experience:
        data = cls._load_experience_data(directory)
        # Select batch:
        if len(data) < config.NN_BATCH_SIZE:
            raise Exception(f"Experience Buffer too small. Data size "
                            f"{len(data)}. Needed {config.NN_BATCH_SIZE}")
        data_batch = np.random.choice(data, config.NN_BATCH_SIZE)
        print("DATA_BATCH :", len(data_batch), data_batch[0])
        states, next_states = cls.get_states_and_next_states(data_batch)
        # Create Team Model:
        tm = cls(states=np.array(states), next_states=np.array(next_states))
        # Save Team model:
        tm_file = os.path.join(directory, config.TEAM_MODEL_FORMAT)
        tm.save_model(tm_file)
        return tm

    @classmethod
    def load_model(cls, file_path: str):
        print("[Team Model] Load model")
        with open(file_path, "rb") as fp:
            team_model: TeamModel = pickle.load(fp)
        return team_model

    def save_model(self, file_path: str):
        print("[Team Model] Saving model")
        with open(file_path, "wb") as fp:
            pickle.dump(self, fp)
            
    def similarity(self, transition: Transition):
        """
        returns: the similarity between model and transition.
        The nearest to zero, the similar it is.
        """
        state = transition.obs
        state = state[np.newaxis, :]
        # Next State:
        next_state = transition.new_obs
        next_state = next_state[np.newaxis, :]

        nearest_state_idx = self.model.query(state, return_distance=False)[0][0]
        next_nearest_state = self.next_states[nearest_state_idx]
        
        sim = next_state - next_nearest_state
        return abs(np.linalg.norm(sim))
