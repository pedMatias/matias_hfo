import random

import numpy as np
from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP, SHOOT, PASS, MOVE

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from agents.dqn_v1.features.discrete_features import \
    DiscFeatures1Teammate
from agents.utils import get_angle


class Actions:
    """ This class uniforms Move and Dribble actions. It allows agent to only
    have to select between 10 actions, instead of 18 actions
    """
    action_w_ball = ["KICK_TO_GOAL", "DRIBBLE_UP", "DRIBBLE_DOWN",
                     "DRIBBLE_LEFT", "DRIBBLE_RIGHT"]
    shoot_possible_coord = [np.array([0.83, -0.18]), np.array([0.83, -0.09]),
                            np.array([0.83, 0]), np.array([0.83, 0.09]),
                            np.array([0.83, 0.18])]
    num_actions = len(action_w_ball)
    action_num_episodes = 15
    
    def get_num_actions(self):
        return self.num_actions
    
    def map_action_to_str(self, action_idx: int) -> str:
        return self.action_w_ball[action_idx]
    
    def dribble_to_pos(self, pos: tuple, features: DiscFeatures1Teammate,
                       game_interface: HFOAttackingPlayer):
        """ The agent keeps dribbling until reach the position expected """
        def check_valid_pos(pos_tuple: tuple):
            for pos_aux in pos_tuple:
                try:
                    num_digits = len(str(pos_aux).split(".")[1])
                    if num_digits >= 2:
                        return False
                except IndexError:
                    pass
            return True
        
        if check_valid_pos(pos) is False:
            raise Exception("Initial positions invalid. Initial positions "
                            "should be a float with 1 digit or less")
        curr_pos = features.get_pos_tuple(round_ndigits=1)
        while pos != curr_pos:
            hfo_action = (DRIBBLE_TO, pos[0], pos[1])
            status, observation = game_interface.step(hfo_action,
                                                      features.has_ball())
            # Update features:
            features.update_features(observation)
            curr_pos = features.get_pos_tuple(round_ndigits=1)
    
    def move_to_pos(self, pos: tuple, features: DiscFeatures1Teammate,
                    game_interface: HFOAttackingPlayer):
        """ The agent keeps moving until reach the position expected """
        curr_pos = features.get_pos_tuple(round_ndigits=1)
        while pos != curr_pos:
            hfo_action = (MOVE_TO, pos[0], pos[1])
            status, observation = game_interface.step(hfo_action,
                                                      features.has_ball())
            # Update features:
            features.update_features(observation)
            curr_pos = features.get_pos_tuple(round_ndigits=1)
    
    def kick_to_pos(self, pos: tuple, features: DiscFeatures1Teammate,
                    game_interface: HFOAttackingPlayer):
        """ The agent kicks to position expected """
        hfo_action = (KICK_TO, pos[0], pos[1], 2)
        status, observation = game_interface.step(hfo_action,
                                                  features.has_ball())
        # Update features:
        features.update_features(observation)

    def shoot_ball(self, game_interface: HFOAttackingPlayer,
                   features: DiscFeatures1Teammate):
        """ Tries to shoot, if it fail, kicks to goal randomly """
        # Get best shoot angle:
        angles = []
        for goal_pos in self.shoot_possible_coord:
            angles.append(get_angle(goalie=features.goalie_coord,
                                    player=features.agent_coord,
                                    point=goal_pos))
        idx = int(np.argmax(np.array(angles)))
        best_shoot_coord = self.shoot_possible_coord[idx]
        # print(" BEST SHOOT COORD: {}".format(best_shoot_coord))
        # Action parameters:
        hfo_action = (KICK_TO, best_shoot_coord[0], best_shoot_coord[1], 2)
        # Step game:
        _, obs = game_interface.step(hfo_action, features.has_ball())
        # Update features:
        features.update_features(obs)
        return game_interface.get_game_status(), \
            game_interface.get_observation_array()

    def move_agent(self, action_name, game_interface: HFOAttackingPlayer,
                   features: DiscFeatures1Teammate):
        """ Agent Moves/Dribbles in a specific direction """
        
        # Get Movement type:
        action = DRIBBLE_TO
        
        if "UP" in action_name:
            action = (action, features.agent_coord[0], - 0.9)
        elif "DOWN" in action_name:
            action = (action, features.agent_coord[0], 0.9)
        elif "LEFT" in action_name:
            action = (action, -0.8, features.agent_coord[1])
        elif "RIGHT" in action_name:
            action = (action, 0.8, features.agent_coord[1])
        else:
            raise ValueError("ACTION NAME is WRONG")
        
        attempts = 0
        while game_interface.in_game() and attempts < self.action_num_episodes:
            status, observation = game_interface.step(action, features.has_ball())
            features.update_features(observation)
            attempts += 1
        return game_interface.get_game_status(), \
            game_interface.get_observation_array()

    def do_nothing(self, game_interface: HFOAttackingPlayer,
                   features: DiscFeatures1Teammate):
        action = (NOOP,)
        status, observation = game_interface.step(action, features.has_ball())
        return status, observation
    
    def no_ball_action(self, game_interface: HFOAttackingPlayer,
                       features: DiscFeatures1Teammate) -> int:
        action = (MOVE,)
        status, observation = game_interface.step(action, features.has_ball())
        features.update_features(observation)
        return status

    def execute_action(self, action_idx: int,
                       game_interface: HFOAttackingPlayer,
                       features: DiscFeatures1Teammate):
        """ Receiving the idx of the action, the agent executes it and
        returns the game status """
        action_name = self.map_action_to_str(action_idx)
        # KICK/SHOOT to goal
        if action_name == "KICK_TO_GOAL":
            status, observation = self.shoot_ball(game_interface, features)
        # MOVE/DRIBBLE
        elif "MOVE" in action_name or "DRIBBLE" in action_name:
            status, observation = self.move_agent(action_name, game_interface,
                                                  features)
        # DO NOTHING
        elif action_name == "NOOP":
            status, observation = self.do_nothing(game_interface, features)
        else:
            raise ValueError("Action Wrong name")
        # Update Features:
        features.update_features(observation)
        return status
