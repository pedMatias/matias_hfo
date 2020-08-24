import numpy as np
from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP, DRIBBLE, PASS, MOVE, \
    GO_TO_BALL

from agents.plastic_v1.actions.base import BaseActions
from agents.plastic_v1.base.hfo_attacking_player import HFOAttackingPlayer
from agents.plastic_v1.features.plastic_features import PlasticFeatures
from agents.utils import get_angle, get_opposite_vector


class Actions(BaseActions):
    name = "plasticSimplexActions"
    # ACTIONS:
    ACTIONS_WITHOUT_BALL = ["MOVE"]
    ACTIONS_WITH_BALL = ["SHOOT", "DRIBBLE"]
    
    NUM_DRIBBLE_STEPS = 4
    NUM_MOVE_STEPS = 10
    NUM_STOP_STEPS = 2

    # SHOOT POSSIBLE POS:
    shoot_possible_coord = [np.array([0.83, -0.17]), np.array([0.83, 0]),
                            np.array([0.83, 0.17])]
    
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
    
    def dribble_action(self, num_rep: int = NUM_DRIBBLE_STEPS):
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
    
    def move_action(self, num_rep: int = 1):
        status = 0
        observation = []
        action = (MOVE,)
        attempts = 0
        while (attempts <= num_rep or not self.features.has_ball()) and \
                self.game_interface.in_game():
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            attempts += 1
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
            elif action_name == "DRIBBLE":
                status, _ = self.dribble_action(self.NUM_DRIBBLE_STEPS)
            elif "PASS" in action_name:
                _, teammate_id = action_name.split("PASS")
                status, _ = self.pass_ball(int(teammate_id))
            else:
                correct_action = False
                status, _ = self.do_nothing(self.NUM_STOP_STEPS)
        else:
            if verbose:
                if action_name in self.ACTIONS_WITHOUT_BALL:
                    print(f"[N] ACTION NAME -> {action_name}")
                else:
                    print(f"[N] WRONG ACTION NAME -> {action_name}")
                    
            if action_name == "MOVE":
                status, _ = self.move_action(self.NUM_MOVE_STEPS)
            else:
                correct_action = False
                status, _ = self.do_nothing(self.NUM_STOP_STEPS)
        return status, correct_action
