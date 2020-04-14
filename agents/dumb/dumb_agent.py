import random
import time

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, MOVE, SHOOT, \
    DRIBBLE, SERVER_DOWN, QUIT, KICK_TO, REORIENT, DRIBBLE_TO, NOOP, TURN

from environement_features.discrete_features import \
    DiscreteHighLevelFeatures
from settings import CONFIG_DIR


if __name__ == '__main__':
    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=6000,
                        config_dir=CONFIG_DIR)
    for i in range(1):
        status = IN_GAME
        score = 0
        NUM_TEAMMATES = 0
        NUM_OPPONENTS = 1
        observation = hfo.getState()
        env = DiscreteHighLevelFeatures(num_team=NUM_TEAMMATES,
                                        num_op=NUM_OPPONENTS)
        counter_moves = 0
        counter_kicks = 0
        while status == IN_GAME:
            if counter_moves < 5:
                print("DRIBBLE")
                hfo.act(DRIBBLE_TO, -0.9, -0.8)
                counter_moves += 1
            elif counter_kicks < 2:
                hfo.act(KICK_TO, 0.9, 0, 3)
                counter_kicks += 1
            else:
                hfo.act(NOOP)
            hfo.step()
        """
            
            if bool(env.has_ball(observation)) is False:
                hfo.act(MOVE)
                print(" >>> Move")
            else:
                if random.random() < 0.8:
                    hfo.act(DRIBBLE)
                    print(" >>> Dribble")
                else:
                    hfo.act(SHOOT)
                    print(" >>> Shoot")
            status = hfo.step()
            if status == SERVER_DOWN:
                hfo.act(QUIT)
                break
            observation = hfo.getState()
            print(" ::: Observation: ", observation)
        """
