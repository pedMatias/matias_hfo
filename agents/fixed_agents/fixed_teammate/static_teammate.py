#!/usr/bin/hfo_env python3
# encoding utf-8
import numpy as np
import math
import argparse

from hfo import MOVE_TO, NOOP, GO_TO_BALL, MOVE, SHOOT, PASS, KICK_TO

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features_1teammate_v1 import \
    DiscreteFeatures1TeammateV1
import settings

ACTIONS = {"MOVE_UP": (MOVE_TO, -0.75, -0.2),
           "MOVE_DOWN": (MOVE_TO, -0.75, 0.2),
           "NOOP": NOOP}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=2)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--wait_for_teammate', type=bool, default=True)
    
    args = parser.parse_args()
    agent_id = args.id
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    port = args.port
    
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(agent_id=agent_id, port=port,
                                       num_opponents=num_op,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    print("<< STATIC AGENT ID - {} >>".format(hfo_interface.hfo.getUnum()))
    
    # Get number of features and actions
    features_manager = DiscreteFeatures1TeammateV1(num_op=num_op,
                                                   num_team=num_team)
    
    for i in range(num_episodes):
        observation = hfo_interface.reset()
        # Update environment features:
        features_manager.update_features(observation)
        while hfo_interface.in_game():
            if features_manager.has_ball():
                hfo_action = (KICK_TO, -0.2, 0, 1.5)
            else:
                hfo_action = (NOOP,)
            
            status, observation = hfo_interface.step(hfo_action,
                                                     features_manager.has_ball())
            
            # Update environment features:
            features_manager._encapsulate_data(observation)
