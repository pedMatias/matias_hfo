#!/usr/bin/hfo_env python3
# encoding utf-8
import numpy as np
import math
import argparse

from hfo import MOVE_TO, NOOP, GO_TO_BALL, DRIBBLE_TO, SHOOT, PASS, KICK_TO

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features_1teammate_v1 import \
    DiscreteFeatures1TeammateV1
import settings

ACTIONS = {"MOVE_TO_TOP_CORNER": (MOVE_TO, 0.4, -0.3),
           "MOVE_TO_BOTTOM_CORNER": (MOVE_TO, 0.4, 0.3),
           "NOOP": NOOP}

DRIBBLE_SHORT = 10  # MOVES
DRIBBLE_LONG = 20  # MOVES

SHORT_KICK_SPEED = 1.5
LONG_KICK_SPEED = 2.6


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--wait_for_teammate', type=bool, default=True)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--port', type=int, default=6000)
    
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    wait_for_teammate = args.wait_for_teammate
    num_episodes = args.num_episodes
    port = args.port
    
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(num_opponents=num_op, port=port,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    uniform_id = hfo_interface.hfo.getUnum()
    teammate_id = 7 if uniform_id == 11 else 11
    print("<< Start FIXED AGENT ID {} >> wait_for_teammate={}; "
          "teammate_id={}".format(uniform_id, wait_for_teammate, teammate_id))
    
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
                if aux_counter == 50:
                    # print("Still waiting. Will start playing")
                    break
        
        # Update environment features:
        features_manager.update_features(observation)
        attempts_to_shoot = 0
        while hfo_interface.in_game():
            # Agent Has ball
            if features_manager.has_ball():
                if features_manager.teammate_further_from_goal():
                    # Goal region:
                    if features_manager.agent.x_pos > 0.3:
                        if attempts_to_shoot > 4:
                            # Failed to kick four times
                            print("Failed to kick four times")
                            hfo_action = (DRIBBLE_TO, 0.45, 0)
                        else:
                            hfo_action = (SHOOT,)
                    else:
                        hfo_action = (DRIBBLE_TO, 0.4, 0)
                else:
                    if not features_manager.agent.last_action_succ:
                        print("Failed to pass ball")
                        hfo_action = (KICK_TO,
                                      features_manager.teammate_coord[0] + 0.05,
                                      features_manager.teammate_coord[1],
                                      2)
                    else:
                        hfo_action = (PASS, teammate_id)
                # print("ACTION = {}".format(hfo_interface.hfo.actionToString(
                #     hfo_action[0])))
            # Teammate has ball
            elif features_manager.teammate_has_ball():
                # Teammate on top position:
                if features_manager.teammate_coord[1] < 0:
                    hfo_action = ACTIONS["MOVE_TO_BOTTOM_CORNER"]
                # Teammate on Bottom position:
                else:
                    hfo_action = ACTIONS["MOVE_TO_TOP_CORNER"]
            # No ball
            else:
                if features_manager.teammate_further_from_ball():
                    hfo_action = (GO_TO_BALL,)
                else:
                    # Move to near corner:
                    if features_manager.agent_coord[1] > 0:
                        hfo_action = ACTIONS["MOVE_TO_BOTTOM_CORNER"]
                    # Move to near corner:
                    else:
                        hfo_action = ACTIONS["MOVE_TO_TOP_CORNER"]
            
            if hfo_action[0] == SHOOT:
                attempts_to_shoot += 1
            else:
                attempts_to_shoot = 0
            
            status, observation = hfo_interface.step(hfo_action,
                                                     features_manager.has_ball())
            
            # Update environment features:
            features_manager.update_features(observation)
