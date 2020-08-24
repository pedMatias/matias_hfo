from abc import ABC, abstractmethod

import numpy as np
from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP, GO_TO_BALL

from agents.offline_plastic_v1.base.hfo_attacking_player import HFOAttackingPlayer
from agents.offline_plastic_v1.features.plastic_features import PlasticFeatures
from agents.utils import get_angle


class BaseActions(ABC):
    # ACTIONS:
    ACTIONS_WITHOUT_BALL = []
    ACTIONS_WITH_BALL = []
    
    # SHOOT POSSIBLE POS:
    shoot_possible_coord = [np.array([0.83, -0.17]), np.array([0.83, 0]),
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

    def move_to_ball(self, num_rep: int = 1):
        """ Move towards the ball  """
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (GO_TO_BALL,)
            status, observation = self.game_interface.step(action)
            
            self.features.update_features(observation)
            #a_coord: np.ndarray = self.features.get_agent_coord()
            #t_coord: np.ndarray = self.features.get_teammate_coord()
            #ball_coord: np.ndarray = self.features.get_ball_coord()
            #
            #if abs(np.linalg.norm(t_coord - ball_coord)) <= 0.1 and \
            #        abs(np.linalg.norm(a_coord - t_coord)) <= 0.2 and \
            #        abs(np.linalg.norm(a_coord - ball_coord)) > 0.1:
            #    return status, observation
            
            attempts += 1
        return status, observation

    def move_to_goal(self, num_rep: int = 1):
        """ Move towards the opposing goal  """
        goal_coord = np.array([0.6, 0])
        status = 0
        observation = []
        attempts = 0
        while self.game_interface.in_game() and attempts < num_rep:
            action = (MOVE_TO, goal_coord[0], goal_coord[1])
            status, observation = self.game_interface.step(action)
            self.features.update_features(observation)
            
            a_coord: np.ndarray = self.features.get_agent_coord()
            if abs(np.linalg.norm(a_coord - goal_coord)) <= 0.15:
                break
                
            attempts += 1
        return status, observation

    def move_to_pos(self, pos: tuple):
        """ The agent keeps moving until reach the position expected """
        pos = (round(pos[0], 2), round(pos[1], 2))
        curr_pos = self.features.get_pos_tuple(round_ndigits=2)
        while abs(pos[0] - curr_pos[0]) > 0.04 and \
                abs(pos[1] - curr_pos[1]) > 0.04 and \
                self.game_interface.in_game():
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

    def discrete_move_agent(self, action_name):
        """ Agent Moves/Dribbles in a discrete form for x number of steps"""
        num_steps = 10
        
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
        while self.game_interface.in_game() and attempts < num_steps:
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
    
    @abstractmethod
    def execute_action(self, action_idx: int, verbose: bool = False) -> \
            (int, bool):
        """ Receiving the idx of the action, the agent executes it and
        returns the game status """
        raise NotImplementedError()
