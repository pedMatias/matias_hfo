import random
import argparse

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, MOVE, SHOOT, \
    DRIBBLE, SERVER_DOWN, QUIT, MOVE_TO, INTERCEPT, DRIBBLE_TO, KICK_TO, NOOP

from environement_features.discrete_features_v2 import DiscreteFeaturesV2
from settings import CONFIG_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)

    args = parser.parse_args()
    port = args.port
    
    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=port,
                        config_dir=CONFIG_DIR)
    for i in range(1):
        status = IN_GAME
        score = 0
        NUM_TEAMMATES = 0
        NUM_OPPONENTS = 1
        observation = hfo.getState()
        env = DiscreteFeaturesV2(num_team=NUM_TEAMMATES, num_op=NUM_OPPONENTS)
        went_to_the_corner = False
        ep = 0
        print("NEW GAME:")
        for i in range(4):
            print("New game:")
            print("Status: ",hfo.step())
            status = IN_GAME
            while status == IN_GAME:
                print("waiting observation")
                observation = hfo.getState()
                env.update_features(observation)
                pos_tuple = env.get_pos_tuple()
                print("waiting action")
                if ep < 10:
                    hfo.act(DRIBBLE_TO, -0.7, 0)
                elif ep < 20:
                    hfo.act(DRIBBLE_TO, 0.4, 0)
                elif ep == 20:
                    print("SHOOT")
                    # hfo.act(KICK_TO, 0.9, 0, 2)
                    hfo.act(SHOOT)
                else:
                    hfo.act(SHOOT)
                print("waiting step")
                status = hfo.step()
                ep += 1
                if status == SERVER_DOWN:
                    hfo.act(QUIT)
                    break
        hfo.act(QUIT)