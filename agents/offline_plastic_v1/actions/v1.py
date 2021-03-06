import random

import numpy as np
from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP, DRIBBLE, PASS, MOVE, \
    GO_TO_BALL

from agents.plastic_v1.base.hfo_attacking_player import HFOAttackingPlayer
from agents.plastic_v1.features.plastic_features import PlasticFeatures
from agents.utils import get_angle, get_opposite_vector


class Actions:
    # ACTIONS:
    ACTIONS_WITHOUT_BALL = ["STAY", "MOVE_TO_BALL", "MOVE_TO_GOAL",
                            "MOVE_TO_NEAR_TEAM", "MOVE_FROM_NEAR_TEAM",
                            "MOVE_TO_NEAR_OP", "MOVE_FROM_NEAR_OP"]
    # 7
    ACTIONS_WITH_BALL = ["SHOOT", "SHORT_DRIBBLE", "LONG_DRIBBLE"]
    
    NUM_SHORT_REP = 5
    NUM_LONG_REP = 20
    # SHOOT POSSIBLE POS:
    shoot_possible_coord = [np.array([0.83, -0.17]), np.array([0.83, -0.09]),
                            np.array([0.83, 0]), np.array([0.83, 0.09]),
                            np.array([0.83, 0.17])]
    
    def __init__(self, num_team: int, features: PlasticFeatures,
                 game_interface: HFOAttackingPlayer):
        self.num_teammates = num_team
        self.features = features
        self.game_interface = game_interface
        
        self.actions = []
        # Actions without ball:
        self.actions += self.ACTIONS_WITHOUT_BALL
        # Actions with ball:
        self.actions += self.ACTIONS_WITH_BALL
        for idx in range(num_team):
            aux_action_name = "PASS" + str(idx)
            self.actions.append(aux_action_name)
            self.ACTIONS_WITH_BALL.append(aux_action_name)
    
        self.num_actions = len(self.actions)
    
        print(f"[ACTIONS] num_team={num_team}, num-action={self.num_actions}")
    
    def get_num_actions(self):
        return self.num_actions
    
    def dribble_to_pos(self, pos: tuple, stop: bool = False):
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
        while pos != curr_pos and self.game_interface.in_game():
            hfo_action = (DRIBBLE_TO, pos[0], pos[1])
            status, observation = self.game_interface.step(hfo_action)
            # Update self.features:
            self.features.update_features(observation)
            curr_pos = self.features.get_pos_tuple(round_ndigits=1)
        # stop agent:
        if stop:
            rep = 0
            while rep <= 5 and self.game_interface.in_game():
                hfo_action = (DRIBBLE_TO, pos[0], pos[1])
                status, observation = self.game_interface.step(hfo_action)
                # Update self.features:
                self.features.update_features(observation)
                rep += 1
    
    def move_to_pos(self, pos: tuple):
        """ The agent keeps moving until reach the position expected """
        pos = (round(pos[0], 2), round(pos[1], 2))
        curr_pos = self.features.get_pos_tuple(round_ndigits=2)
        while pos != curr_pos and self.game_interface.in_game():
            hfo_action = (MOVE_TO, pos[0], pos[1])
            status, observation = self.game_interface.step(hfo_action)
            # Update self.features:
            self.features.update_features(observation)
            curr_pos = self.features.get_pos_tuple(round_ndigits=2)
    
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
        hfo_action = (KICK_TO, best_shoot_coord[0], best_shoot_coord[1], 2.3)
        # Step game:
        status, obs = self.game_interface.step(hfo_action)
        # Update self.features:
        self.features.update_features(obs)
        return status, obs

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

    def do_nothing(self, num_rep: int = 1):
        status = 0
        observation = []
        attempts = 0
        while attempts <= num_rep and self.game_interface.in_game():
            status, observation = self.game_interface.step(NOOP)
            self.features.update_features(observation)
            attempts += 1
        return status, observation
    
    def dribble_action(self, num_rep: int = 1):
        status = 0
        observation = []
        attempts = 0
        while attempts <= num_rep and self.game_interface.in_game():
            if self.features.has_ball():
                action = DRIBBLE
            else:
                action = GO_TO_BALL
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        while not self.features.has_ball() and self.game_interface.in_game():
            action = GO_TO_BALL
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
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
                hfo_action = (KICK_TO, x_pos, y_pos, 1.7)
                status, obs = self.game_interface.step(hfo_action)
                self.features.update_features(obs)
                break
            else:
                hfo_action = (PASS, uniform)
                status, obs = self.game_interface.step(hfo_action)
                self.features.update_features(obs)
                attempts += 1
        return status, obs
    
    def move_to_ball(self, num_rep: int = 1):
        """ Move towards the ball  """
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (GO_TO_BALL,)
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        return status, observation
    
    def move_to_goal(self, num_rep: int = 1):
        """ Move towards the opposing goal  """
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (MOVE_TO, 0.7, 0)
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        return status, observation
    
    def move_to_nearest_teammate(self, num_rep: int = 1):
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            t_coord = self.features.get_nearest_teammate_coord()
            action = (MOVE_TO, t_coord[0], t_coord[1])
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        return status, observation
    
    def move_away_from_nearest_teammate(self, num_rep: int = 1):
        a_coord: np.ndarray = self.features.get_agent_coord()
        t_coord: np.ndarray = self.features.get_nearest_teammate_coord()
        op_vector = get_opposite_vector(a_coord, t_coord)
        # Coordinates:
        x_pos = a_coord[0] + op_vector[0]
        y_pos = a_coord[1] + op_vector[1]
        if abs(x_pos) > 0.8:
            x_pos = 0.8 if x_pos > 0 else -0.8
        if abs(y_pos) > 0.8:
            y_pos = 0.8 if y_pos > 0 else -0.8
        
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (MOVE_TO, x_pos, y_pos)
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        return status, observation

    def move_to_nearest_opponent(self, num_rep: int = 1):
        o_coord = self.features.get_nearest_opponent_coord()
        
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (MOVE_TO, o_coord[0], o_coord[1])
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        return status, observation

    def move_away_from_nearest_opponent(self, num_rep: int = 1):
        a_coord: np.ndarray = self.features.get_agent_coord()
        o_coord: np.ndarray = self.features.get_nearest_opponent_coord()
        op_vector = get_opposite_vector(a_coord, o_coord)
        # Coordinates:
        x_pos = a_coord[0] + op_vector[0]
        y_pos = a_coord[1] + op_vector[1]
        if abs(x_pos) > 0.8:
            x_pos = 0.8 if x_pos > 0 else -0.8
        if abs(y_pos) > 0.8:
            y_pos = 0.8 if y_pos > 0 else -0.8
        
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (MOVE_TO, x_pos, y_pos)
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
        return status, observation

    def execute_action(self, action_idx: int, verbose: bool = False) -> \
            (int, bool):
        """ Receiving the idx of the action, the agent executes it and
        returns the game status """
        # Check action_idx:
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"[Actions] action_idx invalid {action_idx}")

        action_name = self.actions[action_idx]
        correct_action = True
        
        if self.features.has_ball():
            if verbose:
                if action_name in self.ACTIONS_WITH_BALL:
                    print(f"[Y] ACTION NAME -> {action_name}")
                else:
                    print(f"[Y] WRONG ACTION NAME -> {action_name}")
            if action_name == "SHOOT":
                status, _ = self.best_shoot_ball()
            elif action_name == "SHORT_DRIBBLE":
                status, _ = self.dribble_action(self.NUM_SHORT_REP)
            elif action_name == "LONG_DRIBBLE":
                status, _ = self.dribble_action(self.NUM_LONG_REP)
            elif "PASS" in action_name:
                _, teammate_id = action_name.split("PASS")
                status, _ = self.pass_ball(int(teammate_id))
            else:
                correct_action = False
                status, _ = self.do_nothing(self.NUM_SHORT_REP)
        else:
            if verbose:
                if action_name in self.ACTIONS_WITHOUT_BALL:
                    print(f"[N] ACTION NAME -> {action_name}")
                else:
                    print(f"[N] WRONG ACTION NAME -> {action_name}")
            if action_name == "STAY":
                status, _ = self.do_nothing(self.NUM_SHORT_REP)
            elif action_name == "MOVE_TO_BALL":
                status, _ = self.move_to_ball(self.NUM_SHORT_REP)
            elif action_name == "MOVE_TO_GOAL":
                status, _ = self.move_to_goal(self.NUM_SHORT_REP)
            elif action_name == "MOVE_TO_NEAR_TEAM":
                status, _ = self.move_to_nearest_teammate(
                    self.NUM_SHORT_REP)
            elif action_name == "MOVE_FROM_NEAR_TEAM":
                status, _ = self.move_away_from_nearest_teammate(
                    self.NUM_SHORT_REP)
            elif action_name == "MOVE_TO_NEAR_OP":
                status, _ = self.move_to_nearest_opponent(
                    self.NUM_SHORT_REP)
            elif action_name == "MOVE_FROM_NEAR_OP":
                status, _ = self.move_away_from_nearest_opponent(
                    self.NUM_SHORT_REP)
            else:
                correct_action = False
                status, _ = self.do_nothing(self.NUM_SHORT_REP)
        return status, correct_action
