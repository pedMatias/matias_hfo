import random

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, MOVE, SHOOT, \
    DRIBBLE, SERVER_DOWN, QUIT

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
        while status == IN_GAME:
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
