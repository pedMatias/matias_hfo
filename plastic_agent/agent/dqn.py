import numpy as np
import random

from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

from plastic_agent import config

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
    def __init__(self, model: Sequential):
        self.model = model

    @classmethod
    def create(cls, num_features: int, num_actions: int,
               learning_rate: float = 0.01):
        print("[DQN] Creating")
        model = Sequential()
        in_dim = num_features
        # Hidden Layers:
        for out_dim, act_funct in config.DQN_LAYERS:
            model.add(Dense(out_dim, input_dim=in_dim, activation=act_funct))
            in_dim = out_dim
        # Output Layer:
        model.add(Dense(num_actions, activation='linear'))
        # Compile Model:
        model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        return cls(model)
    
    @classmethod
    def load(cls, load_file: str, learning_rate: float = None):
        print("[DQN] Loading")
        model = load_model(load_file)
        if learning_rate:
            K.set_value(model.optimizer.learning_rate, learning_rate)
        return cls(model)
    
    def fit(self, x: list, y: list, verbose: int = 0, epochs: int = 1,
            batch_size: int = MINIBATCH_SIZE):
        # Fit on all samples as one batch, log only on terminal state
        history = self.model.fit(np.array(x), np.array(y), epochs=epochs,
                                 shuffle=True, verbose=verbose,
                                 batch_size=batch_size)
        try:
            loss = history.history["loss"]
        except KeyError as e:
            print("Loss not avaiable. History: ", history.history)
            raise e
        return loss
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        state = state[np.newaxis, :]
        qs: np.ndarray = self.model.predict(state)
        return qs
    
    def save_model(self, file_name: str):
        self.model.save(file_name)
