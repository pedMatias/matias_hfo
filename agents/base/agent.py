from abc import ABC, abstractmethod

import numpy as np


# Agent class
class Agent(ABC):
    def __init__(self, num_features: int, num_actions: int,
                 learning_rate: float = 0.01, discount_factor: float = 0.9,
                 epsilon: float = 0.9, final_epsilon: float = 0.2):
        # Attributes:
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.trained_eps = 0
        
        # Epsilons:
        self.final_epsilon = final_epsilon
        self.EPSILON_VALUES = [round(_, 1) for _ in np.arange(
            epsilon, final_epsilon, -0.1)]
        self.EPSILON_VALUES.append(final_epsilon)

        # An array with last n steps for training
        self.replay_buffer = list()
        
        # Metrics:
        self.num_explorations = 0
        self.num_exploitations = 0
    
    def store_transition(self, curr_st: np.ndarray, action_idx: int,
                         reward: int, new_st: np.ndarray, done: int):
        """ Adds step's data to a memory replay array
        (observation space, action, reward, new observation space, done) """
        transition = [curr_st, action_idx, reward, new_st, done]
        self.replay_buffer.append(transition)
    
    def exploit_actions(self, state: np.ndarray, verbose: bool = False) -> int:
        # print("Exploiting action")
        q_predict = self.get_qs(state)
        max_list = np.where(q_predict == q_predict.max())
        if len(max_list[0]) > 1:
            action = np.random.choice(max_list[0])
        else:
            action = np.argmax(q_predict)
        if verbose:
            print("Q values {} -> {}".format(q_predict, int(action)))
        return int(action)

    def explore_actions(self):
        # print("Exploring action")
        random_action = np.random.randint(0, self.num_actions)
        return random_action

    def act(self, state: np.ndarray, verbose: bool = False):
        if np.random.random() < self.epsilon:  # Explore
            if verbose:
                print("[ACT] Explored")
            self.num_explorations += 1
            return self.explore_actions()
        else:  # Exploit
            if verbose:
                print("[ACT] Exploit")
            self.num_exploitations += 1
            return self.exploit_actions(state)
    
    def restart(self, num_total_episodes: int):
        # Increment num episodes trained
        self.trained_eps += 1
        # Epsilon:
        epsilon_idx = int((self.trained_eps * len(self.EPSILON_VALUES)) /
                          num_total_episodes)
        if epsilon_idx >= len(self.EPSILON_VALUES):
            self.epsilon = self.EPSILON_VALUES[-1]
        else:
            self.epsilon = self.EPSILON_VALUES[int(epsilon_idx)]
    
    @abstractmethod
    def create_model(self, num_features: int, num_actions: int):
        pass
    
    @abstractmethod
    def train(self, terminal_state):
        raise NotImplementedError()
    
    @abstractmethod
    def get_qs(self, state: np.ndarray):
        """ Queries main network for Q values given current observation
        space """
        raise NotImplementedError()
    
    @abstractmethod
    def save_model(self, file_name: str):
        raise NotImplementedError()
    
    @abstractmethod
    def load_model(self, load_file: str):
        raise NotImplementedError()