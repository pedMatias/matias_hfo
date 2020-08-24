import numpy as np
import yaml
from torch.nn.functional import softmax
from collections import deque

from agents.plastic_dqn import config
from models.ReplayMemoryModel import ReplayMemoryModel
from environment.PursuitState import PursuitState
from models.FeedForwardNetwork import FeedForwardNetwork


class TeamModel:

    def __init__(self, num_teammates: int, trainable: bool, num_features):
        self._name = "teammates model"
        self._replay_memory = deque(maxlen=config.REPLAY_MEMORY_SIZE)
        self._replay_batch_size = config.REPLAY_MIN_BATCH
        self._trainable = trainable
        self._input_layers = self.parse_layer_blueprints(
            config.TEAMMATE_MODEL_LAYERS)

        # #################### #
        # Auxiliary Structures #
        # #################### #

        self.num_teammates = num_teammates
        self.teammate_names = [f"teammate_{i + 1}" for i in range(num_teammates)]
        self.features = (self.num_teammates + 1) * 2
        self.total_outputs = 4 * self.num_teammates
        self.prediction_cache = dict()

        # ################ #
        # Hyper Parameters #
        # ################ #

        self.learning_rate = config.LEARNING_RATE

        # ##### #
        # Model #
        # ##### #

        self.models = [
            FeedForwardNetwork(
                num_inputs=self.features,
                num_outputs=4,
                layers=self._input_layers,
                learning_rate=self.learning_rate,
                cuda=True
            ) for _ in range(num_teammates)
        ]
    
    def create_model(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam
        import tensorflow as tf
        tf.compat.v1.set_random_seed(0)
    
        model = Sequential()
        model.add(Dense(128, input_dim=num_features, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    @staticmethod
    def parse_layer_blueprints(layer_blueprints):
        layers = []
        for units, activation in layer_blueprints:
            layers.append((int(units), activation))
        return layers

    # ############## #
    # MAIN INTERFACE #
    # ############## #

    def simulate_teammates_actions(self, state):
        policies = self.policies(state)
        actions = [teammate_policy.argmax() for teammate_policy in policies]
        return actions

    # ########## #
    # PREDICTION #
    # ########## #

    def policies(self, state):
        policies = self.get_from_cache(state) if self.in_cache(state) else self.uncached_prediction(state)
        return policies

    def predict_teammate_policy(self, teammate_id, state):
        x = PursuitState.features_relative_agent(state, teammate_id)
        x.reshape(1, -1)
        model = self.models[teammate_id - 1]
        scores = model.predict(x)
        policy = softmax(scores, dim=0).numpy()
        policy /= policy.sum()
        return policy

    def in_cache(self, state):
        return state in self.prediction_cache

    def get_from_cache(self, state):
        policies = self.prediction_cache[state]
        return policies

    def uncached_prediction(self, state):
        policies = [self.predict_teammate_policy(i + 1, state) for i in range(self.num_teammates)]
        self.prediction_cache[state] = policies
        return policies

    # ######## #
    # TRAINING #
    # ######## #

    def fit_batch(self, batch, verbose=False):
        info = {}
        self.prediction_cache.clear()
        X, Y = self.prepare_individual_batches(batch)
        for i, model in enumerate(self.models):
            if verbose: print(f"Training Teammate {i + 1}", flush=True)
            losses, _, accuracies = model.update(X[i], Y[i], epochs=1, batch_size=self._replay_batch_size, verbose=verbose)
            info[f"teammate {i}"] = {}
            info[f"teammate {i}"]["training loss"] = losses[-1]
            info[f"teammate {i}"]["training accuracy"] = accuracies[-1]
        return info

    def prepare_individual_batches(self, batch):

        m = len(batch)
        F = self.features

        N = len(self.models)

        X = [np.zeros((m, F)) for _ in range(N)]
        Y = [np.zeros(m) for _ in range(N)]

        for i, datapoint in enumerate(batch):
            state, joint_actions, reward, next_state, terminal = datapoint
            for t in range(N):
                teammate = t + 1
                teammate_action = joint_actions[teammate]
                x = PursuitState.features_relative_agent(state, teammate)
                X[t][i] = x
                Y[t][i] = teammate_action

        return X, Y

    # ########### #
    # PERSISTENCE #
    # ########### #

    def save(self, directory):
        super().save(directory)
        for i, model in enumerate(self.models):
            model.save(f"{directory}/{self.teammate_names[i]}")

    def _load(self, directory):
        for i, model in enumerate(self.models):
            model.load(f"{directory}/{self.teammate_names[i]}")
