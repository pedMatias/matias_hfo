import random

import numpy as np
from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP, DRIBBLE, PASS, MOVE, \
    GO_TO_BALL

from agents.plastic_v0.base.hfo_attacking_player import HFOAttackingPlayer
from agents.plastic_v0.features.plastic_features import PlasticFeatures
from agents.utils import get_angle


class Actions:
    # ACTIONS:
    action_without_ball = ["NOOP", "MOVE_TO_BALL", "MOVE_TO_GOAL",
                           "MOVE_TO_NEAR_TEAM", "MOVE_FROM_NEAR_TEAM",
                           "MOVE_TO_NEAR_OP", "MOVE_FROM_NEAR_OP"]
    ACTIONS_W_BALL_DEFAULT = ["SHOOT", "SHORT_DRIBBLE", "LONG_DRIBBLE"]
    NUM_SHORT_REP = 5
    NUM_LONG_REP = 10
    # SHOOT POSSIBLE POS:
    shoot_possible_coord = [np.array([0.83, -0.17]), np.array([0.83, -0.09]),
                            np.array([0.83, 0]), np.array([0.83, 0.09]),
                            np.array([0.83, 0.17])]
    
    def __init__(self, num_team: int, features: PlasticFeatures,
                 game_interface: HFOAttackingPlayer):
        self.num_teammates = num_team
        self.features = features
        self.game_interface = game_interface
        
        self.action_w_ball = self.ACTIONS_W_BALL_DEFAULT
        for idx in range(num_team):
            self.action_w_ball.append("PASS" + str(idx))
            
        self.num_actions = len(self.action_w_ball)
        # print(f"[ACTIONS] num_team={num_team}, num-action={self.num_actions}")
    
    def get_num_actions(self):
        return self.num_actions
    
    def dribble_to_pos(self, pos: tuple):
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
        curr_pos = self.features.get_pos_tuple(round_ndigits=1)
        while pos != curr_pos:
            hfo_action = (DRIBBLE_TO, pos[0], pos[1])
            status, observation = self.game_interface.step(hfo_action)
            # Update self.features:
            self.features.update_features(observation)
            curr_pos = self.features.get_pos_tuple(round_ndigits=1)
    
    def move_to_pos(self, pos: tuple):
        """ The agent keeps moving until reach the position expected """
        curr_pos = self.features.get_pos_tuple(round_ndigits=1)
        while pos != curr_pos:
            hfo_action = (MOVE_TO, pos[0], pos[1])
            status, observation = self.game_interface.step(hfo_action)
            # Update self.features:
            self.features.update_features(observation)
            curr_pos = self.features.get_pos_tuple(round_ndigits=1)
    
    def kick_to_pos(self, pos: tuple):
        """ The agent kicks to position expected """
        hfo_action = (KICK_TO, pos[0], pos[1], 2)
        status, observation = self.game_interface.step(hfo_action)
        # Update self.features:
        self.features.update_features(observation)

    def best_shoot_ball(self):
        """ Tries to shoot, if it fail, kicks to goal randomly """
        # Get best shoot angle:
        angles = []
        goalie_coord = np.array([self.features.opponents[0].x_pos,
                                 self.features.opponents[0].y_pos])
        player_coord = np.array(self.features.get_pos_tuple())
        for goal_pos in self.shoot_possible_coord:
            angles.append(get_angle(goalie=goalie_coord, player=player_coord,
                                    point=goal_pos))
        idx = int(np.argmax(np.array(angles)))
        best_shoot_coord = self.shoot_possible_coord[idx]
        # Action parameters:
        hfo_action = (KICK_TO, best_shoot_coord[0], best_shoot_coord[1], 2.5)
        # Step game:
        _, obs = self.game_interface.step(hfo_action)
        # Update self.features:
        self.features.update_features(obs)
        return self.game_interface.get_game_status(), \
               self.game_interface.get_observation()

    def disc_move_agent(self, action_name):
        """ Agent Moves/Dribbles in a discrete form """
        
        # Get Movement type:
        action = DRIBBLE_TO
        
        if "UP" in action_name:
            action = (action, self.features.agent.x_pos, - 0.9)
        elif "DOWN" in action_name:
            action = (action, self.features.agent.x_pos, 0.9)
        elif "LEFT" in action_name:
            action = (action, -0.8, self.features.agent.y_pos)
        elif "RIGHT" in action_name:
            action = (action, 0.8, self.features.agent.y_pos)
        else:
            raise ValueError("ACTION NAME is WRONG")
        
        attempts = 0
        while self.game_interface.in_game() and attempts < self.NUM_LONG_REP:
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        return self.game_interface.get_game_status(), \
            self.game_interface.get_observation()

    def do_nothing(self):
        action = (NOOP,)
        status, observation = self.game_interface.step(action)
        return status, observation
    
    def dribble_action(self, num_rep: int):
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts <= num_rep:
            if self.features.has_ball():
                action = (DRIBBLE,)
            else:
                action = (GO_TO_BALL,)
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        return status, observation
    
    def move_action(self):
        action = (MOVE,)
        status, observation = self.game_interface.step(action)
        self.features.update_features(observation)
        return status, observation
    
    def pass_ball(self, teammate_id: int):
        """ Tries to use the PASS action, if it fails, Kicks in the direction
        of the teammate"""
        x_pos = self.features.teammates[teammate_id].x_pos
        y_pos = self.features.teammates[teammate_id].y_pos
        uniform = self.features.teammates[teammate_id].uniform_num

        status = 0
        obs = []
        attempts = 0
        while self.game_interface.in_game() and self.features.has_ball():
            if attempts >= 2:
                hfo_action = (KICK_TO, x_pos, y_pos, 1.5)
                status, obs = self.game_interface.step(hfo_action)
                self.features.update_features(obs)
                break
            else:
                hfo_action = (PASS, uniform)
                status, obs = self.game_interface.step(hfo_action)
                self.features.update_features(obs)
                attempts += 1
        return status, obs

    def execute_action(self, action_idx: int, with_ball: bool):
        """ Receiving the idx of the action, the agent executes it and
        returns the game status """
        if with_ball:
            # Check action_idx:
            if 0 > action_idx >= len(self.action_w_ball):
                raise ValueError(f"[Actions] action_idx invalid {action_idx}")
            # Execute Action:
            action_name = self.action_w_ball[action_idx]
            # print(f"ACTION NAME -> {action_name}")
            if action_name == "SHOOT":
                status, observation = self.best_shoot_ball()
            elif action_name == "SHORT_DRIBBLE":
                status, observation = self.dribble_action(self.NUM_SHORT_REP)
            elif action_name == "LONG_DRIBBLE":
                status, observation = self.dribble_action(self.NUM_LONG_REP)
            else:
                _, teammate_id = action_name.split("PASS")
                status, observation = self.pass_ball(int(teammate_id))
        else:
            status, observation = self.move_action()
        return status, observation
