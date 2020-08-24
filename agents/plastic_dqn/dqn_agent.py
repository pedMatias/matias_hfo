import numpy as np
import random
from collections import deque
from typing import List

import tensorflow as tf

from agents.plastic_dqn.replay_buffer import ReplayBuffer, Transition
from agents.plastic_dqn.dqn import DQN


REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to
# start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
# MODEL_NAME = '2x256'

# For more repetitive results
random.seed(2)
np.random.seed(2)


# Agent class
class DQNAgent:
    def __init__(self, num_features: int, num_actions: int,
                 learning_rate: float = 0.01, discount_factor: float = 0.9,
                 epsilon: float = 0.9, final_epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, tau: float = 0.125,
                 create_model: bool = True):
        
        # Attributes:
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.trained_eps = 0
        
        # Metrics:
        self.num_explorations = 0
        self.num_exploitations = 0

        # Epsilon:
        self.epsilon_min = final_epsilon
        self.epsilon_decay = epsilon_decay

        # Models:
        self.tau = tau
        if create_model:
            self.dqn = DQN(num_features=num_features, num_actions=num_actions,
                           learning_rate=learning_rate)
        else:
            self.dqn = None

        # An array with last n steps for training
        self.replay_buffer = ReplayBuffer(memory_size=REPLAY_MEMORY_SIZE)
    
    def store_transition(self, **kwargs):
        self.replay_buffer.store_transition(**kwargs)
        
    def get_qs(self, state: np.ndarray):
        return self.dqn.predict(state)
    
    def exploit_actions(self, state: np.ndarray, verbose: bool = False) -> int:
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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        
    def fit_batch(self, minibatch: List[Transition], verbose: int = 0,
                  epochs: int = 1) -> list:
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition.obs for transition in minibatch])
        current_qs_list = self.dqn.predict(current_states)
        
        # Get future states from minibatch, then query NN model for Q values
        new_states = np.array([transition.new_obs for transition in minibatch])
        future_qs_list = self.dqn.predict(new_states)
        
        # Now we need to enumerate our batches
        X = []
        y = []
        for idx, transition in enumerate(minibatch):
            # If not a terminal state, get new q from future states, else 0
            # almost like with Q Learning, but we use just part of equation
            if not transition.done:
                max_future_q = max(future_qs_list[idx])
                target_td = transition.reward + (self.discount_factor *
                                                 max_future_q)
                td = target_td
            else:
                td = transition.reward
        
            # Update Q value for given state
            current_qs = current_qs_list[idx]
            current_qs[transition.act] = td
            # current_qs[action] = current_qs[action] + self.learning_rate * td
        
            X.append(transition.obs)
            y.append(current_qs)
    
        # Fit on all samples as one batch, log only on terminal state
        loss = self.dqn.fit(x=X, y=y, epochs=epochs, verbose=verbose)
        return loss

