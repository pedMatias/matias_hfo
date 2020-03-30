import random

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, MOVE, SHOOT, \
    DRIBBLE, SERVER_DOWN, QUIT, MOVE_TO, INTERCEPT

from environement_features.discrete_features_v2 import DiscreteFeaturesV2
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
        env = DiscreteFeaturesV2(num_team=NUM_TEAMMATES, num_op=NUM_OPPONENTS)
        went_to_the_corner = False
        while status == IN_GAME:
            observation = hfo.getState()
            print("\n>>> Oppening angle: ", observation[8], "; Opp distance: ",
                  observation[9], "; Distance to Goal:", observation[6],
                  "; Goal Center Angle:", observation[7])
            env.update_features(observation)
            if observation[5] == 1:
                hfo.act(DRIBBLE)
            else:
                hfo.act(MOVE)
            status = hfo.step()
            if status == SERVER_DOWN:
                hfo.act(QUIT)
                break
