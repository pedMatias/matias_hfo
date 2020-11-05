#!/usr/bin/hfo_env python3
# encoding utf-8
import argparse
import json
import os
import pickle
import random
import time
from collections import namedtuple
from copy import deepcopy
from typing import List

import numpy as np

from plastic_agent import config
from plastic_agent.agent.dqn import DQN
from plastic_agent.player.player import Player
from plastic_agent.agent.replay_buffer import LearnBuffer, Transition
from plastic_agent.hfo_env.game_interface import GameInterface
from plastic_agent.hfo_env.features.plastic import PlasticFeatures
from plastic_agent.hfo_env.actions.plastic import PlasticActions

"""
This module is used to train the model using exploration data
"""

TrainMetrics = namedtuple(
    "TrainMetrics",
    ("losses", "saved_iterations", "num_rep", "epochs", "batch_size",
     "min_batch_size", "learning_rate", "discount_factor")
)

EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
MINIBATCH_SIZE = config.MINIBATCH_SIZE
DISCOUNT_FACTOR = config.DQN_DISCOUNT_FACTOR
NUM_MIN_STABLE_TRAINING_EP = config.NUM_MIN_STABLE_TRAINING_EP


class Trainer:
    def __init__(self, num_opponents: int, num_teammates: int, directory: str,
                 step: int):
        # Game Interface:
        self.game_interface = GameInterface(num_opponents=num_opponents,
                                            num_teammates=num_teammates)
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = PlasticActions(num_team=num_teammates,
                                      features=self.features,
                                      game_interface=self.game_interface)
        # DQN:
        self.dqn = DQN.create(
            num_features=self.features.get_num_features(),
            num_actions=self.actions.get_num_actions(),
            learning_rate=LEARNING_RATE
        )
        # Attributes:
        self.directory = directory
        self.step = step
        # Metrics:
        self.replay_buffer = list()
        self.saved_iterations = []
        self.losses = []
    
    def _restart_replay_buffer(self):
        self.replay_buffer = list()

    def _save_model(self, model_base: str, iter: int, model: DQN = None,
                    save_as_main_model: bool = False):
        if save_as_main_model:
            main_model_file = model_base
            main_model_file = os.path.join(self.directory, main_model_file)
            model = self.dqn if model is None else model
            model.save_model(file_name=main_model_file)
        # Save iteration:
        model_file = f"{model_base}.{len(self.saved_iterations)}"
        model_file = os.path.join(self.directory, model_file)
        # Model:
        model = self.dqn if model is None else model
        model.save_model(file_name=model_file)
        self.saved_iterations.append(iter)
    
    def _fit_batch(self, minibatch: List[Transition], verbose: int = 0,
                   epochs: int = 1) -> list:
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition.obs for transition in minibatch])
        current_qs_list = self.dqn.model.predict(current_states)
    
        # Get future states from minibatch, then query NN model for Q values
        new_states = np.array([transition.new_obs for transition in minibatch])
        future_qs_list = self.dqn.model.predict(new_states)
    
        # Now we need to enumerate our batches
        X = []
        y = []
        for idx, transition in enumerate(minibatch):
            # If not a terminal state, get new q from future states, else 0
            # almost like with Q Learning, but we use just part of equation
            if not transition.done:
                max_future_q = max(future_qs_list[idx])
                td = transition.reward + (DISCOUNT_FACTOR * max_future_q)
            else:
                td = transition.reward
        
            # Update Q value for given state
            current_qs = current_qs_list[idx]
            current_qs[transition.act] = td
            # current_qs[action] = current_qs[action] + self.learning_rate * td
        
            X.append(transition.obs)
            y.append(current_qs)
    
        # Fit on all samples as one batch, log only on terminal state
        loss = self.dqn.fit(np.array(X), np.array(y), epochs=epochs,
                            verbose=verbose, batch_size=MINIBATCH_SIZE)
        return loss

    def _load_learn_buffer(self, data_file: str):
        if os.path.isfile(data_file):
            with open(data_file, "rb") as fp:
                learn_buffer: LearnBuffer = pickle.load(fp)
            data: list = learn_buffer.buffer
            self.replay_buffer += data
        else:
            ValueError(f"Can not find file {data_file}")
    
    def load_experience_from_dir(self, clean_learn_buffer: bool,
                                 verbose=False):
        if clean_learn_buffer:
            self._restart_replay_buffer()
        for prev_step in range(0, self.step + 1):
            data_file = config.EXPERIENCE_BUFFER_FORMAT.format(step=prev_step)
            data_file = os.path.join(self.directory, data_file)
            self._load_learn_buffer(data_file)
        if verbose:
            print(f"\n[TRAIN : Step {self.step}] "
                  f"DATA LEN={len(self.replay_buffer)};\n")

    def train_model(self, verbose: bool = False):
        print(f"[train_model: {self.step}] Started")
        start_time = time.time()
        
        num_rep = len(self.replay_buffer) // BATCH_SIZE
        model_base = config.MODEL_FILE_FORMAT.format(step=self.step)
        for i in range(num_rep):
            print(f"::: {i}/{num_rep}")
            # Early save model:
            if i == (num_rep // 2):
                self._save_model(model_base=model_base, iter=i)
            # Train:
            train_data = random.sample(self.replay_buffer, BATCH_SIZE)
            loss = self._fit_batch(train_data, verbose=0, epochs=EPOCHS)
            self.losses.append(sum(loss) / len(loss))
            
        # Trained Min number of iterations
        self._save_model(model_base=model_base, iter=num_rep,
                         save_as_main_model=True)
        
        models = []
        new_losses = []
        for i in range(num_rep, num_rep + NUM_MIN_STABLE_TRAINING_EP):
            print(f"::: {i}/{num_rep + NUM_MIN_STABLE_TRAINING_EP}")
            # Train:
            train_data = random.sample(self.replay_buffer, BATCH_SIZE)
            loss = self._fit_batch(train_data, verbose=0, epochs=EPOCHS)
            avr_loss = sum(loss) / len(loss)
            # Save model:
            models.append(deepcopy(self.dqn))
            new_losses.append(avr_loss)
            self.losses.append(avr_loss)
        
        # Save the new model with lower loss:
        i = new_losses.index(min(new_losses))
        self._save_model(model_base=model_base, iter=i, model=models[i])
        
        duration = (time.time() - start_time) // 60  # Minutes
        print(f"[train_model: {self.step}] Ended. Took {duration} minutes")
        
        return TrainMetrics(
            losses=self.losses,
            saved_iterations=self.saved_iterations,
            num_rep=num_rep+NUM_MIN_STABLE_TRAINING_EP,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            min_batch_size=MINIBATCH_SIZE,
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_teammates', type=int, default=0, required=True)
    parser.add_argument('--num_opponents', type=int, default=0, required=True)
    parser.add_argument('--team_name', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--dir', type=str, required=True)
    
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    team_name = args.team_name
    step = args.step
    directory = args.dir
    
    # Start Trainer:
    trainer = Trainer(num_teammates=num_team, num_opponents=num_op, step=step,
                      directory=directory)
    trainer.load_experience_from_dir(clean_learn_buffer=True, verbose=True)
    metrics: TrainMetrics = trainer.train_model(verbose=True)

    metrics_file = f"train_metrics.{step}.json"
    metrics_file = os.path.join(directory, metrics_file)
    with open(metrics_file, 'w+') as fp:
        json.dump(metrics._asdict(), fp)
    
    print("\n!!!!!!!!! Train End !!!!!!!!!!!!\n\n")
