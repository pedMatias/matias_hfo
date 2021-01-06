import numpy as np
from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP, DRIBBLE, PASS, MOVE, \
    GO_TO_BALL

from agents.offline_plastic_v2.actions.base import BaseActions
from agents.offline_plastic_v2.base.hfo_attacking_player import HFOAttackingPlayer
from agents.offline_plastic_v2.features.plastic_features import PlasticFeatures
from agents.utils import get_angle, get_opposite_vector


class Actions(BaseActions):
    name = "plasticActions"
    # ACTIONS:
    ACTIONS_WITHOUT_BALL = ["NOOP", "MOVE_TO_BALL", "MOVE_TO_GOAL",
                            "MOVE_TO_NEAR_TEAM", "MOVE_FROM_NEAR_TEAM",
                            "MOVE_TO_NEAR_OP", "MOVE_FROM_NEAR_OP"]
    ACTIONS_WITH_BALL = ["SHOOT", "SHORT_DRIBBLE", "LONG_DRIBBLE"]
    
    NUM_SHORT_DRIBBLE_STEPS = 4
    NUM_LONG_DRIBBLE_STEPS = 15
    NUM_GO_TO_BALL_STEPS = 10
    NUM_MOVE_STEPS = 4
    NUM_NOOP_STEPS = 4
    NUM_STOP_STEPS = 1
    
    def __init__(self, num_team: int, features: PlasticFeatures,
                 game_interface: HFOAttackingPlayer):
        super().__init__(num_team, features, game_interface)

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
    
    def dribble_action(self, num_rep: int = 1, long: bool = False):
        status = 0
        observation = []
        attempts = 0
        while attempts <= num_rep and self.game_interface.in_game():
            if not self.features.has_ball():
                action = GO_TO_BALL
            else:
                action = DRIBBLE
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
            
        while not self.features.has_ball() and self.game_interface.in_game():
            action = GO_TO_BALL
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
        return status, observation

    def move_to_nearest_teammate(self, num_rep: int = 1):
        t_coord: np.ndarray = self.features.get_teammate_coord()
        
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (MOVE_TO, t_coord[0], t_coord[1])
            status, observation = self.game_interface.step(action)
            
            self.features.update_features(observation)
            
            dist_to_teammate = self.features.t_coord - self.features.a_coord
            if abs(np.linalg.norm(dist_to_teammate)) <= 0.2:
                break
            attempts += 1
        return status, observation

    def move_away_from_nearest_teammate(self, num_rep: int = 1):
        a_coord: np.ndarray = self.features.get_agent_coord()
        t_coord: np.ndarray = self.features.get_teammate_coord()
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
        
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (MOVE_TO, self.features.near_op_coord[0],
                      self.features.near_op_coord[1])
            status, observation = self.game_interface.step(action)
            
            self.features.update_features(observation)
            if abs(np.linalg.norm(self.features.near_op_coord -
                                  self.features.a_coord)) <= 0.2:
                break
                
            attempts += 1
        return status, observation

    def move_away_from_nearest_opponent(self, num_rep: int = 1):
        op_vector = get_opposite_vector(self.features.a_coord,
                                        self.features.near_op_coord)
        # Coordinates:
        x_pos = self.features.a_coord[0] + op_vector[0]
        y_pos = self.features.a_coord[1] + op_vector[1]
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
    
    def pass_ball(self, teammate_id: int):
        """ Tries to use the PASS action, if it fails, Kicks in the direction
        of the teammate"""
        uniform = self.features.teammates[teammate_id].uniform_num

        status = 0
        obs = []
        attempts = 0
        while self.game_interface.in_game() and self.features.has_ball():
            if attempts >= 2:
                x_pos = self.features.t_coord[0]
                y_pos = self.features.t_coord[1]
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

    def execute_action(self, action_idx: int, verbose: bool = False) -> \
            (int, bool, bool):
        """ Receiving the idx of the action, the agent executes it and
        returns the game status """
        # Check action_idx:
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"[Actions] action_idx invalid {action_idx}")

        action_name = self.actions[action_idx]
        correct_action = True
        passed_ball_succ = False
        
        if self.features.has_ball():
            if verbose:
                if action_name in self.ACTIONS_WITH_BALL:
                    print(f"[Correct Action] {action_name};")
                else:
                    print(f"[Wrong   Action] {action_name};")
                    
            if action_name == "SHOOT":
                status, _ = self.best_shoot_ball()
            elif action_name == "SHORT_DRIBBLE":
                status, _ = self.dribble_action(self.NUM_SHORT_DRIBBLE_STEPS)
            elif action_name == "LONG_DRIBBLE":
                status, _ = self.dribble_action(self.NUM_LONG_DRIBBLE_STEPS,
                                                long=True)
            elif "PASS" in action_name:
                _, teammate_id = action_name.split("PASS")
                status, _ = self.pass_ball(int(teammate_id))
                passed_ball_succ = True
            else:
                correct_action = False
                status, _ = self.do_nothing(self.NUM_STOP_STEPS)
        else:
            if verbose:
                if action_name in self.ACTIONS_WITHOUT_BALL:
                    print(f"[Correct Action] {action_name};")
                else:
                    print(f"[Wrong   Action] {action_name};")

            if action_name == "NOOP":
                status, observation = self.do_nothing(self.NUM_NOOP_STEPS)
            elif action_name == "MOVE_TO_BALL":
                status, observation = self.move_to_ball(self.NUM_GO_TO_BALL_STEPS)
            elif action_name == "MOVE_TO_GOAL":
                status, observation = self.move_to_goal(self.NUM_MOVE_STEPS)
            elif action_name == "MOVE_TO_NEAR_TEAM":
                status, observation = self.move_to_nearest_teammate(
                    self.NUM_MOVE_STEPS)
            elif action_name == "MOVE_FROM_NEAR_TEAM":
                status, observation = self.move_away_from_nearest_teammate(
                    self.NUM_MOVE_STEPS)
            elif action_name == "MOVE_TO_NEAR_OP":
                status, observation = self.move_to_nearest_opponent(
                    self.NUM_MOVE_STEPS)
            elif action_name == "MOVE_FROM_NEAR_OP":
                status, observation = self.move_away_from_nearest_opponent(
                    self.NUM_MOVE_STEPS)
            else:
                correct_action = False
                status, _ = self.do_nothing(self.NUM_STOP_STEPS)
        return status, correct_action, passed_ball_succ
