import pickle
import os

import numpy as np
from sklearn.neighbors import KDTree

from plastic_agent import config
from plastic_agent.agent.replay_buffer import Transition, ExperienceBuffer


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
    def _load_experience_data(directory: str, team_name: str):
        replay_buffer_file = config.REPLAY_BUFFER_FORMAT.format(
            base_dir=directory, team_name=team_name)
        if not os.path.isfile(replay_buffer_file):
            exp_buffer = ExperienceBuffer.create_by_merge_files(
                directory, team_name=team_name)
        else:
            exp_buffer = ExperienceBuffer.load(replay_buffer_file)
        print("File ", replay_buffer_file)
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
    def create_and_save(cls, directory: str, team_name: str):
        # Load previous experience:
        data = cls._load_experience_data(directory, team_name)
        states, next_states = cls.get_states_and_next_states(data)
        # Create Team Model:
        tm = cls(states=np.array(states), next_states=np.array(next_states))
        # Save Team model:
        tm_file = config.TEAM_MODEL_FORMAT.format(base_dir=directory,
                                                  team_name=team_name)
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
