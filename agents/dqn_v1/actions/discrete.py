import random

from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP, SHOOT, PASS, MOVE

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from agents.dqn_v1.features.discrete_features import \
    DiscFeatures1Teammate


class DiscreteActions:
    """ This class uniforms Move and Dribble actions. It allows agent to only
    have to select between 10 actions, instead of 18 actions
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
    
    def get_num_actions(self):
        return self.num_actions
    
    def map_action_to_str(self, action_idx: int, has_ball: bool) -> str:
        if has_ball:
            return self.action_w_ball[action_idx]
        else:
            return self.action_w_out_ball[action_idx]
    
    def dribble_to_pos(self, pos: tuple, features: DiscFeatures1Teammate,
                       game_interface: HFOAttackingPlayer):
        """ The agent keeps dribbling until reach the position expected """
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
        attempts = 0
        # while game_interface.in_game() and features.has_ball():
        #     if attempts > 3:
        #         break
        #     elif attempts == 3:
        #         # Failed to kick four times
        #         # print("Failed to SHOOT 3 times. WILL KICK")
        #         y = 0  # TODO random.choice([0.17, 0, -0.17])
        #         hfo_action = (KICK_TO, 0.9, y, 2)
        #     else:
        #         hfo_action = (SHOOT,)
        #     _, obs = game_interface.step(hfo_action, features.has_ball())
        #     features.update_features(obs)
        #     attempts += 1
        hfo_action = (KICK_TO, 0.9, 0, 2)
        _, obs = game_interface.step(hfo_action, features.has_ball())
        features.update_features(obs)
        return game_interface.get_game_status(), \
            game_interface.get_observation_array()
    
    def pass_ball(self, game_interface: HFOAttackingPlayer,
                  features: DiscFeatures1Teammate):
        """ Tries to use the PASS action, if it fails, Kicks in the direction
        of the teammate"""
        attempts = 0
        while game_interface.in_game() and features.has_ball():
            if attempts > 2:
                break
            elif attempts == 2:
                # Failed to pass 2 times
                # print("Failed to PASS two times. WILL KICK")
                hfo_action = (KICK_TO, features.teammate_coord[0],
                              features.teammate_coord[1], 1.5)
            else:
                hfo_action = (PASS, 11)
            _, obs = game_interface.step(hfo_action, features.has_ball())
            features.update_features(obs)
            attempts += 1
        return game_interface.get_game_status(), \
            game_interface.get_observation_array()

    def move_agent(self, action_name, game_interface: HFOAttackingPlayer,
                   features: DiscFeatures1Teammate):
        """ Agent Moves/Dribbles in a specific direction """
        # print("move_agent!")
        if "SHORT" in action_name:
            num_repetitions = 10
        elif "LONG" in action_name:
            num_repetitions = 20
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
        while game_interface.in_game() and attempts < num_repetitions:
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
        action_name = self.map_action_to_str(action_idx, features.has_ball())
        # KICK/SHOOT to goal
        if action_name == "KICK_TO_GOAL":
            status, observation = self.shoot_ball(game_interface, features)
        # PASS ball to teammate
        elif action_name == "PASS":
            status, observation = self.pass_ball(game_interface, features)
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
