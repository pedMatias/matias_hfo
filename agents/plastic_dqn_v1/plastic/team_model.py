import pickle
from typing import List

import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict

from agents.plastic_dqn_v1.agent.dqn_agent import Transition


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
    
    @classmethod
    def create_and_save(cls, directory: str, team_name: str):
        replay_buffer_file = config.REPLAY_BUFFER_FORMAT.format(
            base_dir=dir, team_name=team_name)
        replay_buffer = ExperienceBuffer.load(replay_buffer_file)
        print("File ", replay_buffer_file)
        print("Replay Buffer: ", len(replay_buffer.replay_memory),
              replay_buffer.replay_memory[0])
    
    @classmethod
    def create_model(cls, data: List[Transition]):
        print("[Team Model] Creating model")
        states = []
        next_states = []
        for transiction in data:
            states.append(transiction.obs)
            next_states.append(transiction.new_obs)
        return cls(states=np.array(states),
                   next_states=np.array(next_states))

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
        raise ValueError()
        state = transition.obs
        state = state[np.newaxis, :]
        # Next State:
        next_state = transition.new_obs
        next_state = next_state[np.newaxis, :]

        nearest_state_idx = self.model.query(state, return_distance=False)[0][0]
        print("nearest_state_idx: ", nearest_state_idx)
        next_nearest_state = self.next_states[nearest_state_idx]
        
        sim = next_state - next_nearest_state
        return abs(np.linalg.norm(sim))
