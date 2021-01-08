import os
import pickle
import random
import time

import numpy as np

from multi_agents import config
from multi_agents.knn.knn_model import KNeighbors
from multi_agents.dqn_agent.replay_buffer import Transition, ExperienceBuffer


class TeamModel:
    """ Saves an KDTree with all the initial states of any transition, and
    also a list with all the Transitions """
    
    def __init__(self, states: np.ndarray, next_states: np.ndarray):

        # KDTree:
        print("[Team Model: KDTree] Creating KDTree")
        self.model: KNeighbors = KNeighbors(k=1)
        self.model.fit(X=states, y=np.array(range(len(states))))
        
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
    def get_states_and_next_states(data: np.ndarray) -> (np.ndarray, np.ndarray):
        states = []
        next_states = []
        for transiction in data:
            states.append(transiction.obs)
            next_states.append(transiction.new_obs)
        return np.array(states), np.array(next_states)
    
    @classmethod
    def set_up_and_save(cls, directory: str):
        # Load previous experience:
        data = cls._load_experience_data(directory)
        # Select batch:
        if len(data) < config.NN_BATCH_SIZE:
            raise Exception(f"Experience Buffer too small. Data size "
                            f"{len(data)}. Needed {config.NN_BATCH_SIZE}")
        data = np.random.choice(data, config.NN_BATCH_SIZE)
        print("DATA_BATCH :", len(data), data[0])
        states, next_states = cls.get_states_and_next_states(data)
        # Team Model Needed data:
        team_model_data = np.array([states, next_states])
        # Create team Model, to see if it is working fine:
        tm = cls(states, next_states)
        # Save Team model:
        tm_file = os.path.join(directory, config.TEAM_MODEL_FORMAT)
        tm.save_needed_data(tm_file, team_model_data)
        tm.small_test(states, next_states)
        return tm

    @classmethod
    def load_model(cls, file_path: str):
        print("[Team Model] Load model")
        with open(file_path, "rb") as fp:
            team_model: TeamModel = pickle.load(fp)
        return team_model

    @classmethod
    def load_model_from_data(cls, file_path: str):
        print("[Team Model] Load model")
        with open(file_path, "rb") as fp:
            team_model_data: np.ndarray = pickle.load(fp)
        return cls(team_model_data[0], team_model_data[1])
    
    def small_test(self, states, next_states):
        ss = time.time()
        curr_sims_list = []
        next_sims_list = []
        for _ in range(100):
            idx = random.choice(range(len(states)))
            # State:
            state = states[idx]
            next_state = next_states[idx]
            # Predict Curr state
            state_idx = self.model.predict(state[np.newaxis, :])
            pred_state = states[state_idx]
            curr_sim = abs(np.linalg.norm(state - pred_state))
            curr_sims_list.append(curr_sim)
            # Predict next state:
            next_sim = self.similarity(state, next_state)
            next_sims_list.append(next_sim)
        average_curr_sim = sum(curr_sims_list) / len(curr_sims_list)
        average_next_sim = sum(next_sims_list) / len(next_sims_list)
        print("Total time: ", time.time() - ss)
        print("Average curr sim: ", average_curr_sim)
        print("Average next sim: ", average_next_sim)
        assert average_curr_sim < 0.1
        assert average_next_sim < 0.1

    def save_needed_data(self, file_path: str, data: np.ndarray):
        print("[Team Model] Saving model")
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)
    
    def predict_next_state(self, state: np.ndarray) -> np.ndarray:
        state = state[np.newaxis, :]
        nearest_state_idx = self.model.predict(state)
        next_nearest_state = self.next_states[nearest_state_idx]
        return next_nearest_state
    
    def similarity(self, state: np.ndarray, next_state: np.ndarray) -> float:
        next_nearest_state = self.predict_next_state(state)
        sim = next_state - next_nearest_state
        return abs(np.linalg.norm(sim))
            
    def transition_similarity(self, transition: Transition):
        """
        returns: the similarity between model and transition.
        The nearest to zero, the similar it is.
        """
        state = transition.obs
        # Next State:
        next_state = transition.new_obs
        return self.similarity(state, next_state)
