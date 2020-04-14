import random
import time

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, MOVE, SHOOT, \
    DRIBBLE, SERVER_DOWN, QUIT, MOVE_TO, DRIBBLE_TO, KICK_TO

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from actions_levels.discrete_actions import DiscreteActions
from environement_features.discrete_features_v2 import DiscreteFeaturesV2


if __name__ == '__main__':
    NUM_TEAMMATES = 0
    NUM_OPPONENTS = 1
    hfo_interface = HFOAttackingPlayer(agent_id=0,
                                       num_opponents=NUM_OPPONENTS,
                                       num_teammates=NUM_TEAMMATES)
    hfo_interface.connect_to_server()
    
    features = DiscreteFeaturesV2(NUM_TEAMMATES, NUM_OPPONENTS)
    actions_manager = DiscreteActions()
    for i in range(1):
        status = IN_GAME
        score = 0
        NUM_TEAMMATES = 0
        NUM_OPPONENTS = 1
        center = False
        while status == IN_GAME:
            features.update_features(hfo_interface.get_state())
            curr_state_id = features.get_state_index()
            features_vect = features.get_features()
            ball_pos = features.ball_position[features_vect[3]]
            print("\nFEATURES: ", features_vect)
            print("BALL_POSITION: ", ball_pos)
            
            if not center:
                for _ in range(10):
                    hfo_interface.step((DRIBBLE_TO, 0, 0), True)
                center = True
                
            agent_pos = features.get_pos_tuple()
            if ball_pos == "Player Has Ball":
                action_name = "KICK_TO_GOAL"
                has_ball = True
            elif ball_pos == "Up":
                action_name = "MOVE_UP"
                has_ball = False
            elif ball_pos == "Down":
                action_name = "MOVE_DOWN"
                has_ball = False
            elif ball_pos == "Right":
                action_name = "MOVE_RIGHT"
                has_ball = False
            elif ball_pos == "Left":
                action_name = "MOVE_LEFT"
                has_ball = False
            else:
                raise ValueError("Ball position unexpected: {}".
                                 format(ball_pos))
            
            print("Action: ", action_name)
            args = actions_manager.get_action_params(agent_pos, action_name)
            status, observation = hfo_interface.step(args, has_ball)
