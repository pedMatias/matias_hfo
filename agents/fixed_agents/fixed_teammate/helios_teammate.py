#!/usr/bin/env python3
# encoding utf-8
import argparse
import numpy as np
import random

from hfo import MOVE_TO, NOOP, MOVE, SHOOT, PASS, DRIBBLE

from agents.utils import get_angle
from agents.offline_plastic_v1.base.hfo_attacking_player import \
    HFOAttackingPlayer
from agents.utils import get_vertices_around_ball
from environement_features.base import BaseHighLevelState
import settings


def teammate_further_from_goal(features_manager: BaseHighLevelState) -> bool:
    """
    @return: 0 if teammate near goal, 1 otherwise
    @rtype: int
    """
    goal_coord = np.array([1, 0])
    agent_coord = np.array([agent_f.x_pos, agent_f.y_pos])
    teammate_coord = np.array([team_f.x_pos, team_f.y_pos])
    if teammate_coord[0] == -2 or teammate_coord[1] == -2:
        return True
    
    team_dist = np.linalg.norm(teammate_coord - goal_coord)
    agent_dist = np.linalg.norm(agent_coord - goal_coord)
    if team_dist < agent_dist:
        return False
    else:
        return True


def set_start_game_conditions(features_manager: BaseHighLevelState,
                              hfo_interface: HFOAttackingPlayer,
                              wait: bool, fixed_position: bool = False):
    ball_pos = [features_manager.agent.ball_x, features_manager.agent.ball_y]
    starting_corners = get_vertices_around_ball(ball_pos)
    start_pos = random.choice(starting_corners)

    aux_counter = 0
    if wait:
        while hfo_interface.in_game():
            msg = hfo_interface.hfo.hear()
            if msg == settings.PLAYER_READY_MSG:
                # print("\n[HELIOS] HEARD MESSAGE!! Start Playing\n")
                break
            else:
                if fixed_position:
                    hfo_action = (MOVE_TO, start_pos[0], start_pos[1])
                    _, observation = hfo_interface.step(hfo_action)
                else:
                    _, observation = hfo_interface.step(NOOP)
                features_manager.update_features(observation)
            aux_counter += 1
            if aux_counter == 120:
                # print("\n[HELIOS] STILL WAITING!! Start Playing\n")
                break
    return


def best_shoot_angle(agent_coord: tuple, opponents: list):
    """ Tries to shoot, if it fail, kicks to goal randomly """
    # Get best shoot angle:
    best_angles = []
    player_coord = np.array(agent_coord)
    goal_limits = [np.array([0.9, -0.2]), np.array([0.9, 0]),
                   np.array([0.9, 0.2])]
    for goal_limit in goal_limits:
        angles = []
        for op_idx in range(0, len(opponents)):
            op_coord = np.array([opponents[op_idx].x_pos,
                                 opponents[op_idx].y_pos])
            angles.append(get_angle(goalie=op_coord, player=player_coord,
                                    point=goal_limit))
        best_angles.append(min(angles))
    # return the best angles avaiable
    return max(best_angles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--wait_for_teammate', type=str, default="true")
    parser.add_argument('--starts_fixed_position', type=str, default="true")
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--port', type=int, default=6000)
    
    args = parser.parse_args()
    print(f"TEAMMATE: {args}")
    num_team = args.num_teammates
    num_op = args.num_opponents
    wait_for_teammate = True if args.wait_for_teammate == "true" else False
    starts_fixed_position = True if args.starts_fixed_position == "true" \
        else False
    num_episodes = args.num_episodes
    port = args.port
    
    verbose = False
    
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(num_opponents=num_op, port=port,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    uniform_id = hfo_interface.hfo.getUnum()
    teammate_id = 7 if uniform_id == 11 else 11
    print("<< Start HELIOS AGENT ID {} >> wait_for_teammate={}; "
          "teammate_id={}".format(uniform_id, wait_for_teammate, teammate_id))
    
    # Get number of features and actions
    features_manager = BaseHighLevelState(num_op=num_op, num_team=num_team)
    
    for i in range(num_episodes):
        observation = hfo_interface.reset()
        features_manager.update_features(observation)
        
        # Set start game conditions:
        set_start_game_conditions(
            features_manager=features_manager,
            hfo_interface=hfo_interface,
            wait=wait_for_teammate,
            fixed_position=starts_fixed_position)
        
        # Update environment features:
        features_manager.update_features(hfo_interface.get_observation())
        
        last_action = -1
        attempts_to_shoot = 0
        last_action_failed = False
        while hfo_interface.in_game():
            # Features:
            agent_f = features_manager.agent
            agent_coord = (agent_f.x_pos, agent_f.y_pos)
            team_f = features_manager.teammates[0]
            team_coord = (team_f.x_pos, team_f.y_pos)
            # Goal angles:
            agent_angle = (best_shoot_angle(agent_coord,
                                            features_manager.opponents))
            team_angle = (best_shoot_angle(team_coord,
                                           features_manager.opponents))
            # Distance to opponent:
            agent_op_dist = agent_f.proximity_op
            team_op_dist = team_f.proximity_op
            # Pass angle
            pass_angle = abs(team_f.pass_angle)
            
            # Agent Has ball
            if features_manager.has_ball():
                # Last action Failed (DRIBBLE):
                if last_action_failed or attempts_to_shoot >= 2:
                    attempts_to_shoot = 0
                    last_action_failed = False
                    hfo_action = DRIBBLE
                        
                # Opponents near agent && good angle pass (PASS):
                elif agent_f.proximity_op < -0.9 and \
                        team_f.proximity_op > -0.7:
                    hfo_action = (PASS, teammate_id)
                    
                # Far from Goal:
                elif agent_f.x_pos < 0.2:
                    # Teammate near Goal, good pass angle and far from op
                    if not teammate_further_from_goal(features_manager) and \
                            team_f.proximity_op > -0.8:
                        hfo_action = (PASS, teammate_id)
                    else:
                        hfo_action = DRIBBLE
                # Near Goal:
                else:
                    # Teammate near GoaL, better goal angle, good pass angle:
                    if team_f.x_pos > 0 and (team_angle - agent_angle > 10) \
                            and team_op_dist > -0.9 and pass_angle > 0:
                        hfo_action = (PASS, teammate_id)
                    # Good shoot angle and opponent far:
                    elif agent_angle > 7 and agent_f.proximity_op > -0.9:
                        hfo_action = SHOOT
                        attempts_to_shoot += 1
                    else:
                        hfo_action = DRIBBLE
            # Teammate has ball
            else:
                # Update attempts to shoot counter:
                attempts_to_shoot = 0
                # Calculate distance between agents:
                agent_pos = np.array([agent_f.x_pos, agent_f.y_pos])
                ball_pos = np.array([agent_f.ball_x, agent_f.ball_y])
                team_pos = np.array([team_f.x_pos, team_f.y_pos])
                agent_ball_dist = np.linalg.norm(agent_pos - ball_pos)
                team_ball_dist = np.linalg.norm(team_pos - ball_pos)
                agents_dist = np.linalg.norm(agent_pos - team_pos)
                
                # If agent closer to ball:
                if agent_ball_dist < team_ball_dist:
                    hfo_action = MOVE
                # If agents very close to each other
                elif agents_dist < 0.4:
                    # Teammate upper side o field
                    if team_pos[1] < 0:
                        hfo_action = (MOVE_TO, 0.4, 0.3)
                    else:
                        hfo_action = (MOVE_TO, 0.4, -0.3)
                # Agents with good distance from each other:
                else:
                    hfo_action = MOVE
            
            if isinstance(hfo_action, tuple):
                last_action = hfo_action[0]
            else:
                last_action = hfo_action
                
            _, observation = hfo_interface.step(hfo_action)
            
            # Update environment features:
            features_manager.update_features(observation)
            last_action_failed = False if \
                agent_f.last_action_succ == 1 else True
