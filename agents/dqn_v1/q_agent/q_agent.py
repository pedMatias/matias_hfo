import json

import numpy as np

from agents.base.agent import Agent


# Agent class
class QAgent(Agent):
    def __init__(self, num_features: int, num_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.9, final_epsilon: float = 0.2):
        
        super().__init__(num_features, num_actions, learning_rate,
                         discount_factor, epsilon, final_epsilon)
        
        # Model:
        self.model: dict = self.create_model(num_features, num_actions)

        # An array with last n steps for training
        self.replay_memory = list()
        self.game_transitions = list()

    def create_model(self, num_features: int, num_actions: int) -> dict:
        """ each entry will be the feature vector converted to string"""
        model = dict()
        return model
    
    # def _rm_actions_without_ball(self, transitions: list):
    #    """ Removes actions withou ball. Although if not any action with
    #    ball, returns the original list
    #    (0:observation, 1:action, 2:reward, 3:new obs, 4:done, 5:has ball)
    #    """
    #    original_trasitions = transitions.copy()
    #    last_reward = transitions[-1][2]
    #    while transitions[-1][5] is False:
    #        transitions = transitions[:-1]
    #        if len(transitions) == 0:
    #            # print("No actions with ball: {}".format(original_trasitions))
    #            return original_trasitions
    #    else:
    #        transitions[-1][2] = last_reward
    #        transitions[-1][4] = True
    #        return transitions
    
    def _observation_vector_to_id(self, obs: np.ndarray) -> str:
        id = ""
        for val in obs:
            id += str(int(val)) + "_"
        id = id[:-1]  # remove _
        return id
    
    def store_episode(self, transitions: list, reward: int = None):
        """ Transitions:
        (0:observation, 1:action, 2:reward, 3:new obs, 4:done)"""
        # transitions = self._rm_actions_without_ball(transitions)
        if reward is not None:
            transitions[-1][2] = reward  # final reward
        transitions[-1][4] = True  # done
        self.replay_memory = []
        for obs, a, r, new_obs, d in transitions:
            st = self._observation_vector_to_id(obs)
            new_st = self._observation_vector_to_id(new_obs)
            self.replay_memory.append((st, a, r, new_st, d))
            self.game_transitions.append((obs, a, r, new_obs, d))

    def train(self, terminal_state):
        """ Trains main network every step during episode """
        minibatch = self.replay_memory

        # Now we need to enumerate our batches
        for idx, (curr_st, action, r, new_st, done) in enumerate(minibatch):
            # print("\n||| ACT={}; R={}; STATE={} \n[OLD] {}".
            #       format(action, r, new_st, self.model_predict(curr_st)))
            
            current_q = self.model_predict(curr_st)
            future_q = self.model_predict(new_st)
            if not done:
                max_future_q = np.max(future_q)
                target_td = r + (self.discount_factor * max_future_q)
                td = target_td - current_q[action]
            else:
                td = r - current_q[action]

            # Update Q value for given state
            current_q[action] = current_q[action] + self.learning_rate * td

            # Fit on all samples as one batch, log only on terminal state
            self.model_fit([curr_st], [current_q])

            # print("[NEW] {}".format(self.model_predict(curr_st)))
        
        # Clean learn buffer
        self.replay_memory = list()
    
    def get_qs(self, state) -> np.ndarray:
        if isinstance(state, np.ndarray):
            state = self._observation_vector_to_id(state)
        if state not in self.model.keys():
            self.model[state] = np.random.rand(self.num_actions)
        return self.model[state]
    
    def model_predict(self, state: str) -> np.ndarray:
        if state not in self.model.keys():
            self.model[state] = np.random.rand(self.num_actions)
        return self.model[state].copy()
    
    def model_fit(self, X, Y):
        for x, y in zip(X, Y):
            self.model[x] = y
    
    def save_model(self, file_name: str):
        new_json_dict = {}
        for key, values in self.model.items():
            new_json_dict[key] = values.tolist()
            
        json_data = json.dumps(new_json_dict)
        with open(file_name, "w+") as f:
            f.write(json_data)

    def save_game_transictions(self, file_name: str):
        game_transictions_arr = np.array(self.game_transitions)
        np.save(file_name, game_transictions_arr)

    def load_model(self, load_file: str):
        with open(load_file) as json_file:
            data = json.load(json_file)

        new_json_dict = {}
        for key, values in data.items():
            new_json_dict[key] = np.array(values)
        return new_json_dict
