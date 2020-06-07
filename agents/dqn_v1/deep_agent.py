import numpy as np
import random
from collections import deque

from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
import tensorflow as tf

from agents.base.agent import Agent


REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to
# start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
# MODEL_NAME = '2x256'

# For more repetitive results
random.seed(2)
np.random.seed(2)
tf.set_random_seed(2)


# Agent class
class DQNAgent(Agent):
    def __init__(self, num_features: int, num_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.9, final_epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, tau: float = 0.125):
        
        super().__init__(num_features, num_actions, learning_rate,
                         discount_factor, epsilon, final_epsilon)
        
        # Epsilon:
        self.epsilon_min = final_epsilon
        self.epsilon_decay = epsilon_decay
        
        # Model:
        self.model = self.create_model(num_features, num_actions)
        self.tau = tau

        # Target network
        # self.target_model = self.create_model(num_features, num_actions)
        # self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def create_model(self, num_features: int, num_actions: int):
        self.model_description = "{}f_24x48x24_{}a".format(num_features,
                                                           num_actions)
        model = Sequential()
        model.add(Dense(32, input_dim=num_features, activation="relu"))
        # model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(32, activation="relu"))
        # model.add(LeakyReLU(alpha=0.1))
        # model.add(Dense(24, activation="relu"))
        # model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def store_transition(self, curr_st: np.ndarray, action_idx: int,
                         reward: int, new_st: np.ndarray, done: int):
        """ Adds step's data to a memory replay array
        (observation space, action, reward, new observation space, done) """
        transition = np.array([curr_st, action_idx, reward, new_st, done])
        self.replay_memory.append(transition)
    
    def store_episode(self, transitions: list, reward: int = None):
        """ Adds step's data to a memory replay array
        (observation space, action, reward, new observation space, done) """
        if len(transitions) == 0:
            return
        else:
            if reward is not None:
                transitions[-1][2] = reward  # final reward
            transitions[-1][4] = True  # done
            for obs, a, r, new_obs, d in transitions:
                self.replay_memory.append((obs, a, r, new_obs, d))

    def train(self, terminal_state: bool):
        """ Trains main network every step during episode """
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MINIBATCH_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        # Get future states from minibatch, then query NN model for Q values
        new_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.model.predict(new_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for idx, (curr_st, action, r, new_st, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, else 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = max(future_qs_list[idx])
                target_td = r + (self.discount_factor * max_future_q)
                td = target_td
                # td = target_td - current_qs_list[idx][action]
            else:
                td = r
                # td = r - current_qs_list[idx][action]

            # Update Q value for given state
            current_qs = current_qs_list[idx]
            current_qs[action] = td
            # current_qs[action] = current_qs[action] + self.learning_rate * td
            
            X.append(curr_st)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)

    def train_from_batch(self, batch: list):
        """ Trains main network using buffer of data """
        # Random shuffle list
        random.shuffle(batch)
        
        while len(batch) >= 100:
            minibatch = batch[-100:]
            batch = batch[:-100]
            
            # Get current states from minibatch, and query NN model for Q values
            current_states = np.array([ep[0] for ep in minibatch])
            current_qs_list = self.model.predict(current_states)
            # Get future states from minibatch, then query NN model for Q values
            new_states = np.array([ep[3] for ep in minibatch])
            future_qs_list = self.model.predict(new_states)
    
            x = []
            y = []
            for idx, (curr_st, action, r, new_st, done) in enumerate(minibatch):
                if r == -1:
                    r = -1000
                elif r == 1:
                    r = 1000
                else:
                    r = -1
                # If not a terminal state, get new q from future states,
                # otherwise set it to 0. Almost like with Q Learning, but we
                # use just part of equation here
                if not done:
                    max_future_q = np.max(future_qs_list[idx])
                    new_q = r + self.discount_factor * max_future_q
                else:
                    new_q = r
    
                # Update Q value for given state
                current_qs = current_qs_list[idx]
                current_qs[action] = new_q
    
                # And append to our training data
                x.append(curr_st)
                y.append(current_qs)
    
            # Fit on all samples as one batch, log only on terminal state
            self.model.fit(np.array(x), np.array(y), batch_size=100, verbose=0)

    def train_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (
                        1 - self.tau)
        self.target_model.set_weights(target_weights)
    
    def restart(self, num_total_episodes: int):
        # Increment num episodes trained
        self.trained_eps += 1
        # Epsilon:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Queries main network for Q values given current observation space
    def get_qs(self, state: np.ndarray):
        state = state[np.newaxis, :]
        qs = self.model.predict(state)
        return qs

    # Queries main network for Q values given current observation space
    def get_target_model_qs(self, state: np.ndarray):
        state = state[np.newaxis, :]
        qs = self.target_model.predict(state)
        return qs
    
    def save_model(self, file_name: str):
        self.model.save(file_name)

    def load_model(self, load_file: str):
        self.model = load_model(load_file)