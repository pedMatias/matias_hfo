import argparse

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, SERVER_DOWN, \
    QUIT, DRIBBLE_TO, DRIBBLE

from environement_features.discrete_features import \
    DiscreteHighLevelFeatures
from settings import CONFIG_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)

    args = parser.parse_args()
    port = args.port
    
    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=port,
                        config_dir=CONFIG_DIR)
    print("Connected")
    for i in range(1):
        status = IN_GAME
        score = 0
        NUM_TEAMMATES = 0
        NUM_OPPONENTS = 2
        observation = hfo.getState()
        env = DiscreteHighLevelFeatures(num_team=NUM_TEAMMATES,
                                        num_op=NUM_OPPONENTS)
        ep_counter = 0
        while status == IN_GAME:
            hfo.act(DRIBBLE)
            # check game status:
            ep_counter += 1
            status = hfo.step()
            env.get_features(hfo.getState())
            print("OP: ", env.agent.proximity_op)
            if status == SERVER_DOWN:
                hfo.act(QUIT)
                break
           
        """
            
            if bool(hfo_env.has_ball(observation)) is False:
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
