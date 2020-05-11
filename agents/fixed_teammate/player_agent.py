#!/usr/bin/env python3
# encoding utf-8
import numpy as np
import math
import argparse

from hfo import MOVE_TO, NOOP, GO_TO_BALL, DRIBBLE_TO, SHOOT, PASS, KICK_TO

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features_1teammate_v1 import \
    DiscreteFeatures1TeammateV1
import settings

ACTIONS = {"MOVE_UP": (MOVE_TO, -0.75, -0.2),
           "MOVE_DOWN": (MOVE_TO, -0.75, 0.2),
           "NOOP": NOOP}

DRIBBLE_SHORT = 10  # MOVES
DRIBBLE_LONG = 20  # MOVES

SHORT_KICK_SPEED = 1.5
LONG_KICK_SPEED = 2.6
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--wait_for_teammate', type=bool, default=True)
    parser.add_argument('--num_episodes', type=int, default=500)
    
    args = parser.parse_args()
    agent_id = args.id
    num_team = args.num_teammates
    num_op = args.num_opponents
    wait_for_teammate = args.wait_for_teammate
    num_episodes = args.num_episodes
    
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(agent_id=agent_id,
                                       num_opponents=num_op,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    print("<< FIXED AGENT ID - {} >>".format(hfo_interface.hfo.getUnum()))
    
    # Get number of features and actions
    features_manager = DiscreteFeatures1TeammateV1(num_op=num_op,
                                                   num_team=num_team)
    
    for i in range(num_episodes):
        observation = hfo_interface.reset()
        aux_counter = 0
        if wait_for_teammate:
            while hfo_interface.in_game():
                msg = hfo_interface.hfo.hear()
                if msg == settings.PLAYER_READY_MSG:
                    break
                else:
                    # if aux_counter % 10 == 0:
                    # print("[FIXED PLAYER] Still Waiting")
                    pass
                status, observation = hfo_interface.step(NOOP, False)
                aux_counter += 1
        # Update environment features:
        features_manager.update_features(observation)
        prev_action = ()
        teammate_id = 7
        while hfo_interface.in_game():
            # Agent Has ball
            if features_manager.has_ball():
                if features_manager.teammate_further_from_goal():
                    # Goal region:
                    if features_manager.get_position_name() == "MID RIGHT":
                        hfo_action = (SHOOT,)
                    else:
                        hfo_action = (DRIBBLE_TO, 0.6, 0)
                else:
                    if not features_manager.agent.last_action_succ:
                        print("Last action Failed")
                        hfo_action = (KICK_TO,
                                      features_manager.teammate_coord[0] + 0.05,
                                      features_manager.teammate_coord[1],
                                      1)
                    else:
                        hfo_action = (PASS, teammate_id)
            # Teammate has ball
            elif features_manager.teammate_has_ball():
                # Top position:
                if features_manager.get_pos_tuple()[1] < 0:
                    hfo_action = (MOVE_TO, 0.5, -0.2)
                # Bottom position:
                else:
                    hfo_action = (MOVE_TO, 0.5, 0.2)
            # No ball
            else:
                if features_manager.teammate_further_from_ball():
                    hfo_action = (GO_TO_BALL,)
                else:
                    # Top position:
                    if features_manager.get_pos_tuple()[1] < 0:
                        hfo_action = (MOVE_TO, 0.5, -0.3)
                    # Bottom position:
                    else:
                        hfo_action = (MOVE_TO, 0.5, 0.3)
            # if prev_action != hfo_action:
            #     print("Action: ", hfo_interface.hfo.actionToString(
            #         hfo_action[0]), hfo_action[1:])
            
            prev_action = hfo_action
            status, observation = hfo_interface.step(hfo_action,
                                                     features_manager.has_ball())
            
            # Update environment features:
            features_manager.update_features(observation)
