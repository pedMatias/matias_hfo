import numpy as np
import random

from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

tf.compat.v1.set_random_seed(0)


# start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
# UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
# MODEL_NAME = '2x256'

# For more repetitive results
random.seed(2)
np.random.seed(2)


# Agent class
class DQN:
    def __init__(self, num_features: int, num_actions: int,
                 learning_rate: float):
        
        # Models:
        self.learning_rate = learning_rate
        self.model = self.create_model(num_features, num_actions)
        
    def create_model(self, num_features: int, num_actions: int):
        """ model.add(LeakyReLU(alpha=0.1))"""
        self.model_description = "{}f_128_128_128_128_{}a".format(
            num_features, num_actions)
        model = Sequential()
        model.add(Dense(128, input_dim=num_features, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def fit(self, x: list, y: list, verbose: int = 0, epochs: int = 1):
        # Fit on all samples as one batch, log only on terminal state
        history = self.model.fit(np.array(x), np.array(y), epochs=epochs,
                                 shuffle=True, verbose=verbose,
                                 batch_size=MINIBATCH_SIZE)
        try:
            loss = history.history["loss"]
        except KeyError as e:
            print("Loss not avaiable. History: ", history.history)
            raise e
        return loss
    
    def predict(self, state: np.ndarray):
        state = state[np.newaxis, :]
        qs = self.model.predict(state)
        return qs
    
    def save_model(self, file_name: str):
        self.model.save(file_name)
    
    # Queries main network for Q values given current observation space
    def load_model(self, load_file: str, learning_rate: float = None):
        self.model = load_model(load_file)
        if learning_rate:
            K.set_value(self.model.optimizer.learning_rate, learning_rate)
