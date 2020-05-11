import random

from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP
import numpy as np

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features_1teammate_v1 import \
    DiscreteFeatures1TeammateV1
import settings


ORIGIN_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                    "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                    "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}


class DiscreteActions1TeammateV1:
    """ This class uniforms Move and Dribble actions. It allows agent to only
    have to select between 6 actions, instead of 10 actions
    """
    action_w_ball = ["KICK_TO_GOAL", "PASS",
                     "LONG_DRIBBLE_UP", "LONG_DRIBBLE_DOWN",
                     "LONG_DRIBBLE_LEFT", "LONG_DRIBBLE_RIGHT",
                     "SHORT_DRIBBLE_UP", "SHORT_DRIBBLE_DOWN",
                     "SHORT_DRIBBLE_LEFT", "SHORT_DRIBBLE_RIGHT"]
    action_w_out_ball = ["NOOP", "NOOP",
                         "LONG_MOVE_UP", "LONG_MOVE_DOWN",
                         "LONG_MOVE_LEFT", "LONG_MOVE_RIGHT",
                         "SHORT_MOVE_UP", "SHORT_MOVE_DOWN",
                         "SHORT_MOVE_LEFT", "SHORT_MOVE_RIGHT"]
    num_actions = len(action_w_ball)
    long_action_num_episodes = 20
    short_action_num_episodes = 10
    
    def __init__(self, origin_pos: tuple = (0, 0)):
        self.origin_pos = origin_pos
    
    def get_num_actions(self):
        return self.num_actions
    
    def map_action_to_str(self, action_idx: int, has_ball: bool) -> str:
        if has_ball:
            return self.action_w_ball[action_idx]
        else:
            return self.action_w_out_ball[action_idx]
    
    def go_to_origin_pos(self) -> tuple:
        return DRIBBLE_TO, self.origin_pos[0], self.origin_pos[1]
    
    def dribble_to_pos(self, pos: tuple) -> tuple:
        return DRIBBLE_TO, pos[0], pos[1]
    
    def map_action_idx_to_hfo_action(self, agent_pos: tuple, has_ball: bool,
                                     action_idx: int,
                                     teammate_pos: np.ndarray) -> (tuple, int):
        if has_ball:
            action_name = self.action_w_ball[action_idx]
        else:
            action_name = self.action_w_out_ball[action_idx]
        return self.get_action_params(agent_pos, action_name, teammate_pos)
    
    def get_action_params(self, position: tuple, action_name: str,
                          teammate_pos: np.ndarray) -> (tuple, int):
        """ Get the action parameters and the number of repetitions"""
        x_pos, y_pos = position
        num_repetitions = 1
        if action_name == "KICK_TO_GOAL":
            return (KICK_TO, 0.9, 0, 2), num_repetitions
        elif action_name == "PASS":
            return (KICK_TO, teammate_pos[0], teammate_pos[1], 1), \
                   num_repetitions
        elif action_name == "NOOP":
            return NOOP, num_repetitions
        else:
            # Get num_repetitions:
            if "SHORT" in action_name:
                num_repetitions = self.short_action_num_episodes
            elif "LONG" in action_name:
                num_repetitions = self.long_action_num_episodes
            else:
                raise ValueError("ACTION NAME is WRONG")
            
            # Get Movement type:
            if "MOVE" in action_name:
                action = MOVE_TO
            elif "DRIBBLE" in action_name:
                action = DRIBBLE_TO
            else:
                raise ValueError("ACTION NAME is WRONG")
            
            if "UP" in action_name:
                return (action, x_pos, - 0.9), num_repetitions
            elif "DOWN" in action_name:
                return (action, x_pos, 0.9), num_repetitions
            elif "LEFT" in action_name:
                return (action, -0.8, y_pos), num_repetitions
            elif "RIGHT" in action_name:
                return (action, 0.8, y_pos), num_repetitions
            else:
                raise ValueError("ACTION NAME is WRONG")
            

def execute_action(action_params: tuple, repetitions:int,
                   game_interface: HFOAttackingPlayer, has_ball: bool):
    rep_counter_aux = 0
    observation = []
    while game_interface.in_game() and rep_counter_aux < repetitions:
        status, observation = game_interface.step(action_params, has_ball)
        rep_counter_aux += 1
    return game_interface.get_game_status(), observation


def go_to_origin_position(game_interface: HFOAttackingPlayer,
                          features: DiscreteFeatures1TeammateV1,
                          actions: DiscreteActions1TeammateV1,
                          random_start: bool = True):
    if random_start:
        pos_name, origin_pos = random.choice(list(ORIGIN_POSITIONS.items()))
    else:
        pos_name = "Fixed start"
        origin_pos = features.get_pos_tuple()
    # print("\nMoving to starting point: {0}".format(pos_name))
    pos = features.get_pos_tuple(round_ndigits=1)
    while origin_pos != pos:
        has_ball = features.has_ball()
        hfo_action: tuple = actions.dribble_to_pos(origin_pos)
        status, observation = game_interface.step(hfo_action, has_ball)
        features.update_features(observation)
        pos = features.get_pos_tuple(round_ndigits=1)
    # Informs the teammate that it is ready to start the game
    game_interface.hfo.say(settings.PLAYER_READY_MSG)
    game_interface.hfo.step()
    game_interface.hfo.say(settings.PLAYER_READY_MSG)
    game_interface.hfo.step()