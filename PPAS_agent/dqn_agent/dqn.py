import numpy as np
import random

from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

from plastic_policy_agent import config

tf.compat.v1.set_random_seed(0)

# For more repetitive results
random.seed(0)
np.random.seed(0)


# Agent class
class DQN:
    def __init__(self, model: Sequential):
        self.model = model

    @classmethod
    def create(cls, num_features: int, num_actions: int, num_teammates: int,
               learning_rate: float = 0.01):
        print("[DQN] Creating")
        model = Sequential()
        # Game mode:
        layers = config.DQN_LAYERS
        # Input Layer:
        out_dim, act_funct = layers["input"]
        model.add(Dense(out_dim, input_dim=num_features, activation=act_funct))
        
        # Hidden Layers:
        for out_dim, act_funct in layers["hidden"]:
            model.add(Dense(out_dim, activation=act_funct))
            
        # Output Layer:
        _, act_funct = layers["output"]
        model.add(Dense(num_actions, activation=act_funct))
        
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
            batch_size: int = config.MINIBATCH_SIZE):
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
