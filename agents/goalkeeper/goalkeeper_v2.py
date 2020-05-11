#!/usr/bin/env python3
# encoding utf-8
import numpy as np
import math
import argparse

from hfo import MOVE_TO, NOOP

from agents.base.hfo_goalkeeper_player import HFOGoalkeeperPlayer
from environement_features.base import BaseHighLevelState
import settings

ACTIONS = {"MOVE_UP": (MOVE_TO, -0.7, -0.15),
           "MOVE_DOWN": (MOVE_TO, -0.7, 0.15),
           "MOVE_MIDDLE": (MOVE_TO, -0.7, 0),
           "NOOP": NOOP}
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=500)
    
    args = parser.parse_args()
    num_episodes = args.num_episodes
    
    # Initialize connection with the HFO server
    hfo_interface = HFOGoalkeeperPlayer()
    hfo_interface.connect_to_server()
    
    # Get number of features and actions
    features_manager = BaseHighLevelState(num_op=1)
    
    for i in range(num_episodes):
        observation = hfo_interface.reset()
        # Update environment features:
        features_manager._encapsulate_data(observation)
        
        while hfo_interface.in_game():
            if features_manager.agent.ball_y <= -0.2:
                hfo_action = ACTIONS["MOVE_UP"]
            elif features_manager.agent.ball_y >= 0.2:
                hfo_action = ACTIONS["MOVE_DOWN"]
            else:
                hfo_action = ACTIONS["MOVE_MIDDLE"]

            status, observation = hfo_interface.step(*hfo_action)
            
            # Update environment features:
            features_manager._encapsulate_data(observation)
