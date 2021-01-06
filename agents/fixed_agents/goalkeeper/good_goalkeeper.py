#!/usr/bin/hfo_env python3
# encoding utf-8
"""
WARNING: goalkeeper coordinates are inverted comparatively to the offense agent
"""
import numpy as np
import math
import argparse

from hfo import MOVE_TO, NOOP, INTERCEPT

from agents.base.hfo_goalkeeper_player import HFOGoalkeeperPlayer
from environement_features.base import BaseHighLevelState
import settings

ACTIONS = {"MOVE_DOWN": (MOVE_TO, -0.75, -0.175),
           "MOVE_UP": (MOVE_TO, -0.75, 0.175),
           "MOVE_MIDDLE": (MOVE_TO, -0.65, 0),
           "INTERCEPT": (INTERCEPT,),
           "NOOP": (NOOP,)}

GOAL_REGION = {"x": [-1, -0.5],
               "y": [-0.4, 0.4]}


def ball_in_goal_region(features: BaseHighLevelState) -> bool:
    if GOAL_REGION['x'][0] <= features.agent.ball_x <= GOAL_REGION['x'][1] and\
            GOAL_REGION['y'][0] <= features.agent.ball_y <= GOAL_REGION['y'][1]:
        return True
    else:
        return False
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--num_offenses', type=int, default=1)
    parser.add_argument('--num_defenses', type=int, default=0)
    parser.add_argument('--port', type=int, default=6000)
    
    args = parser.parse_args()
    num_episodes = args.num_episodes
    num_team = args.num_defenses
    num_op = args.num_offenses
    port = args.port
    
    # Initialize connection with the HFO server
    hfo_interface = HFOGoalkeeperPlayer(port=port)
    hfo_interface.connect_to_server()
    
    # Get number of features and actions
    features_manager = BaseHighLevelState(num_op=num_op, num_team=num_team)
    
    for i in range(num_episodes):
        observation = hfo_interface.reset()
        # Update environment features:
        features_manager._encapsulate_data(observation)
        
        while hfo_interface.in_game():
            if ball_in_goal_region(features_manager):
                hfo_action = ACTIONS["INTERCEPT"]
            elif features_manager.agent.ball_y <= -0.2:
                hfo_action = ACTIONS["MOVE_DOWN"]
            elif features_manager.agent.ball_y >= 0.2:
                hfo_action = ACTIONS["MOVE_UP"]
            else:
                hfo_action = ACTIONS["MOVE_MIDDLE"]

            status, observation = hfo_interface.step(*hfo_action)
            # print("DEF (x={}, y={})".format(features_manager.agent.ball_x,
            #                                 features_manager.agent.ball_y))
            # Update environment features:
            features_manager._encapsulate_data(observation)
