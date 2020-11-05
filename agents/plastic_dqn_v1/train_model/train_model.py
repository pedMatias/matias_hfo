#!/usr/bin/env python3
# encoding utf-8
import argparse
import json
import os
import pickle
import random
from copy import copy
from typing import List
from collections import deque

import numpy as np
from hfo import GOAL, IN_GAME, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS

import settings
from agents.utils import ServerDownError, get_vertices_around_ball
from agents.plastic_dqn_v1.base.hfo_attacking_player import \
    HFOAttackingPlayer
from agents.plastic_dqn_v1.agent.dqn import DQN
from agents.plastic_dqn_v1.agent.replay_buffer import LearnBuffer, Transition
from agents.plastic_dqn_v1.actions.complex import Actions
from agents.plastic_dqn_v1.features.plastic_features import \
    PlasticFeatures
from agents.plastic_dqn_v1 import config
from agents.plastic_dqn_v1.aux import print_transiction, mkdir


"""
This module is used to train the model using exploration data
"""


DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.00025
REPLAY_MEMORY_SIZE = 1_000_000  # How many last steps to keep for model training
NUM_EPOCHS = 10
NUM_TRAIN_REP = 25
NUM_RETRAIN_REP = 10
MINIBATCH_SIZE = 64


class Player:
    def __init__(self, num_opponents: int, num_teammates: int,
                 model_file: str):
        # Game Interface:
        self.game_interface = HFOAttackingPlayer(
            num_opponents=num_opponents,
            num_teammates=num_teammates)
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = Actions(num_team=num_teammates,
                               features=self.features,
                               game_interface=self.game_interface)
        # DQNs:
        self.dqn = DQN.load(load_file=model_file)
        
        # Replay buffer:
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)

    def load_experience(self, dir: str, step: int, team_name: str,
                        verbose: bool = False):
        for prev_step in range(0, step + 1):
            experience_file = config.EXPERIENCE_BUFFER_FORMAT.format(
                team_name=team_name, step=prev_step)
            data_file = os.path.join(dir, experience_file)
            if os.path.isfile(data_file):
                with open(data_file, "rb") as fp:
                    learn_buffer: LearnBuffer = pickle.load(fp)
                    data: list = learn_buffer.buffer
                    if verbose: print(f"Add stage {prev_step} data. "
                                      f"SIZE={len(data)}")
                    self.replay_buffer += data
        if verbose:
            print(f"\n[TRAIN : Step {step}] "
                  f"DATA LEN={len(self.replay_buffer)};\n")

    def fit_batch(self, minibatch: List[Transition], verbose: int = 0,
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
                target_td = transition.reward + (DISCOUNT_FACTOR *
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
        loss = self.dqn.fit(np.array(X), np.array(y), epochs=epochs,
                            verbose=verbose, batch_size=MINIBATCH_SIZE)
        return loss

    def train_model(self, dir: str, team_name: str, step: int,
                    retrain: bool = True, verbose: bool = False):
        # Auxiliar vars:
        saved_iterations = []
        losses = []
        c_num_stable_trains = 0
        
        # Arguments:
        if retrain:
            num_min_stable_trains = 10
            num_rep = NUM_RETRAIN_REP
            file_name = "re_" + config.MODEL_FILE_FORMAT.format(
                team_name=team_name, step=step)
        else:
            num_min_stable_trains = 5
            num_rep = NUM_TRAIN_REP
            file_name = "new_" + config.MODEL_FILE_FORMAT.format(
                team_name=team_name, step=step)
        model_file = os.path.join(dir, file_name)
        for i in range(num_rep):
            # Early stop:
            if c_num_stable_trains >= num_min_stable_trains:
                break
        
            # Train:
            train_data = self.replay_buffer.copy()
            loss = self.fit_batch(train_data, verbose=0, epochs=NUM_EPOCHS)
        
            # Avr Loss:
            avr_loss = sum(loss) / len(loss)

            # Save model:
            if i > 0 and (avr_loss < min(losses) or
                          avr_loss < min(losses[i - 5:i])):
                # Save over last saved:
                if i - 1 in saved_iterations:
                    saved_iterations[-1] = i
                else:
                    saved_iterations.append(i)
                new_model_file = f"{model_file}.{len(saved_iterations)}"
                self.dqn.save_model(file_name=new_model_file)
    
                # Check if loss changed less than 1%
                changed_percentage = (avr_loss * 100) / losses[-1]
                changed_percentage = 100 - changed_percentage
                if 0 < changed_percentage < 3:
                    c_num_stable_trains += 1
                else:
                    c_num_stable_trains = 0
                
                if verbose:
                    print(f"[{i}/{num_rep}] Loss={avr_loss}; "
                          f"Variation={changed_percentage}; SAVED!")
            else:
                if verbose:
                    print(f"[{i}/{num_rep}] Loss={avr_loss}; N-S!")
            # Save Loss:
            losses.append(avr_loss)
        else:
            saved_iterations.append(num_rep)
            new_model_file = model_file + "." + str(len(saved_iterations))
            self.dqn.save_model(file_name=new_model_file)
        return losses, saved_iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--save_all', type=str, default="false")
    parser.add_argument('--team_name', type=str, default=None)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--dir', type=str)

    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    team_name = args.team_name
    step = args.step
    directory = args.dir
    
    # Get previous model:
    if int(step) == 0:
        # Load base model:
        model_file = config.BASE_MODEL_DQN
    else:
        # Beginning of a stage:
        prev_step = step - 1
        # Get model file:
        model_file = config.MODEL_FILE_FORMAT.format(team_name=team_name,
                                                     step=prev_step)
        model_file = os.path.join(directory, model_file)

    print(f"[TRAIN Player: {team_name}; Step {step}] num_t={num_team}; "
          f"num_op={num_op}; dir={directory}; base_model={model_file};")
    
    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op,
                    model_file=model_file)
    
    player.load_experience(dir=directory, team_name=team_name, step=step,
                           verbose=True)
    
    # TRAIN model:
    player.train_model(retrain=True, dir=directory, team_name=team_name,
                       step=step, verbose=True)

    # TRAIN New model:
    player.train_model(retrain=False, dir=directory, team_name=team_name,
                       step=step, verbose=True)
    
    print("\n!!!!!!!!! Train End !!!!!!!!!!!!\n\n")
