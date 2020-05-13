import random
import time

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, MOVE, SHOOT, \
    DRIBBLE, SERVER_DOWN, QUIT, MOVE_TO, DRIBBLE_TO, KICK_TO

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from actions_levels.discrete_actions import DiscreteActions
from environement_features.base import BaseHighLevelState


if __name__ == '__main__':
    NUM_TEAMMATES = 0
    NUM_OPPONENTS = 1
    hfo_interface = HFOAttackingPlayer(agent_id=0,
                                       num_opponents=NUM_OPPONENTS,
                                       num_teammates=NUM_TEAMMATES)
    hfo_interface.connect_to_server()
    
    features = BaseHighLevelState(NUM_TEAMMATES, NUM_OPPONENTS)
    actions_manager = DiscreteActions()
    for i in range(1):
        status = IN_GAME
        center = False
        while status == IN_GAME:
            obs_array = hfo_interface.get_state()
            features._encapsulate_data(obs_array)
            # print("OFF {}".format(obs_array))
            # print("\nOff BALL ({}, {})".format(features.agent.ball_x,
            #                                    features.agent.ball_y))
            # print("Off Agent ({}, {})".format(features.agent.x_pos,
            #                                   features.agent.y_pos))
            hfo_action = (DRIBBLE_TO, 0.6, 0.2)
            has_ball = features.agent.can_kick
            status, observation = hfo_interface.step(hfo_action, has_ball)

