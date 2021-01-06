#!/usr/bin/hfo_env python3
# encoding utf-8
import argparse
import pickle
import random
from copy import copy
from typing import List

import numpy as np
from hfo import GOAL, IN_GAME, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS

import settings
from agents.utils import ServerDownError, get_vertices_around_ball
from agents.plastic_v1.base.hfo_attacking_player import HFOAttackingPlayer
from agents.plastic_v1.deep_agent import DQNAgent, Transition, MINIBATCH_SIZE
from agents.plastic_v1.actions.simplex import Actions
from agents.plastic_v1.features.plastic_features import PlasticFeatures
from agents.plastic_v1.aux import print_transiction

STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                      "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                      "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}


class Player:
    def __init__(self, num_opponents: int, num_teammates: int,
                 port: int = 6000):
        # Game Interface:
        self.game_interface = HFOAttackingPlayer(num_opponents=num_opponents,
                                                 num_teammates=num_teammates,
                                                 port=port)
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = Actions(num_team=num_teammates, features=self.features,
                               game_interface=self.game_interface)
        # Agent instance:
        self.agent = DQNAgent(num_features=self.features.num_features,
                              num_actions=self.actions.get_num_actions(),
                              learning_rate=0.005, discount_factor=0.99,
                              epsilon=1, final_epsilon=0.001,
                              epsilon_decay=0.99997, tau=0.125)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--dir', type=str, default=None)
    
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    directory = args.dir
    
    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op)

    with open(f"{directory}/learn_buffer", "rb") as fp:  # Unpickling
        train_data = pickle.load(fp)
    
    print(f"TRAIN DATA len={len(train_data)} from {directory};\n"
          f"{train_data[0]}")
    
    losses = []
    for i in range(20):
        # Get a minibatch of random samples from memory replay table
        loss = player.agent.fit_batch(train_data, verbose=1)
        player.agent.target_model.set_weights(player.agent.model.get_weights())
        # Loss:
        avr_loss = sum(loss) / len(loss)
        print(f"{i}: Avarage loss {avr_loss}")
        losses.append(avr_loss)
    
    player.agent.save_model(file_name=directory + "/agent_model")
    print("\n!!!!!!!!! AGENT EXIT !!!!!!!!!!!!\n\n")
