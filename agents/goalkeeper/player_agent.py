#!/usr/bin/env python3
# encoding utf-8
import numpy as np
import math
import argparse

from hfo import MOVE_TO, NOOP

from agents.base.hfo_goalkeeper_player import HFOGoalkeeperPlayer
from environement_features.base import BaseHighLevelState
import settings

ACTIONS = {"MOVE_UP": (MOVE_TO, -0.75, -0.2),
           "MOVE_DOWN": (MOVE_TO, -0.75, 0.2),
           "NOOP": NOOP}
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--num_offenses', type=int, default=1)
    parser.add_argument('--num_defenses', type=int, default=0)
    
    args = parser.parse_args()
    num_team = args.num_defenses
    num_op = args.num_offenses
    num_episodes = args.num_episodes
    
    # Initialize connection with the HFO server
    hfo_interface = HFOGoalkeeperPlayer()
    hfo_interface.connect_to_server()
    
    # Get number of features and actions
    features_manager = BaseHighLevelState(num_op=num_op, num_team=num_team)
    
    for i in range(num_episodes):
        observation = hfo_interface.reset()
        # Update environment features:
        features_manager._encapsulate_data(observation)
        
        while hfo_interface.in_game():
            if features_manager.agent.ball_y <= 0:
                hfo_action = ACTIONS["MOVE_UP"]
            else:
                hfo_action = ACTIONS["MOVE_DOWN"]

            status, observation = hfo_interface.step(*hfo_action)
            
            # Update environment features:
            features_manager._encapsulate_data(observation)
