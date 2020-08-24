import numpy as np
import sys
from collections import deque

from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
import time
import random

from agents.base.agent import Agent

# For more repetitive results
random.seed(2)
np.random.seed(2)
tf.set_random_seed(2)


# Agent class
class QAgent(Agent):
    def __init__(self, num_features: int, num_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.9, final_epsilon: float = 0.3):
        
        super().__init__(num_features, num_actions, learning_rate,
                         discount_factor, epsilon, final_epsilon)
        # Learning buffer
        self.transition_buffer = list()

        # Main model
        self.model = self.create_model(num_features, num_actions)

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
   
    def get_qs(self, state: np.ndarray, model=None):
        """ Queries main network for Q values given current observation space"""
        state = state[np.newaxis, :]
        if model is None:
            qs = self.model.predict(state)
        else:
            qs = model.predict(state)
        return qs[0]

    def train(self, goal: bool):
        """ Trains main network every step during episode """

        def remove_mov_without_ball(transitions):
            """ remove movements without ball """
            last_reward = transitions[-1][2]
            has_ball = False
            i = len(transitions) - 1
            while not has_ball:
                if i < 0:  # Never caught the ball
                    return []  # Will not learn
                elif transitions[i][0][6] == 0:  # HAS BALL
                    transitions = transitions[:i + 1]
                    transitions[-1][2] = last_reward
                    transitions[-1][4] = True
                    return transitions
                else:
                    i -= 1
        
        buffer = self.transition_buffer.copy()
        if goal:
            buffer = remove_mov_without_ball(buffer)

        # Now we need to enumerate our batches
        buffer.reverse()
        for idx, (curr_st, action, r, new_st, done) in enumerate(buffer):
            # If not a terminal state, get new q from future states, else 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                future_qs = self.get_qs(state=new_st)
                max_future_q = np.max(future_qs)
                new_q = r + self.discount_factor * max_future_q
            else:
                new_q = r

            # Update Q value for given state
            current_qs = self.get_qs(curr_st)
            current_qs[action] = new_q

            # And append to our training data
            X = curr_st if isinstance(curr_st, np.ndarray) else np.array(curr_st)
            y = current_qs if isinstance(current_qs, np.ndarray) \
                else np.array(current_qs)
            # Convert Format:
            X = X[np.newaxis, :]
            y = y[np.newaxis, :]
            
            # Fit on all samples as one batch, log only on terminal state
            self.model.fit(X, y, verbose=0, shuffle=False)

        # Inc number of trained episodes:
        self.trained_eps += 1
        self.transition_buffer = []

    def save_model(self, file_name: str):
        self.model.save(file_name)

    def load_model(self, load_file: str):
        self.model = load_model(load_file)
