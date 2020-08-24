import numpy as np
import sys

from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random

from agents.base.agent import Agent


REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to  start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'

# For more repetitive results
random.seed(2)
np.random.seed(2)
tf.set_random_seed(2)


# Agent class
class DQNAgent(Agent):
    def __init__(self, num_features: int, num_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.9, final_epsilon: float = 0.2):
        
        super().__init__(num_features, num_actions, learning_rate,
                         discount_factor, epsilon, final_epsilon)

        # Target network
        self.target_model = self.create_model(num_features, num_actions)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self, num_features: int, num_actions: int):
        model = Sequential()
        model.add(Dense(64, input_shape=(num_features,)))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy'])
        return model
    
    def store_transition(self, curr_st: np.ndarray, action_idx: int,
                         reward: int, new_st: np.ndarray, done: int):
        """ Adds step's data to a memory replay array
        (observation space, action, reward, new observation space, done) """
        transition = np.array([curr_st, action_idx, reward, new_st, done])
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        """ Trains main network every step during episode """
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        # Get future states from minibatch, then query NN model for Q values
        new_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for idx, (curr_st, action, r, new_st, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, else 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[idx])
                new_q = r + self.discount_factor * max_future_q
            else:
                new_q = r

            # Update Q value for given state
            current_qs = current_qs_list[idx]
            current_qs[action] = new_q

            # And append to our training data
            X.append(curr_st)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space
    def get_qs(self, state: np.ndarray):
        state = state[np.newaxis, :]
        qs = self.model.predict(state)
        return qs
    
    def save_model(self, file_name: str):
        self.model.save(file_name)

    def load_model(self, load_file: str):
        self.model = load_model(load_file)