import argparse

from hfo import IN_GAME, GOAL, DRIBBLE, KICK, GO_TO_BALL, DRIBBLE_TO, NOOP

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from agents.dqn_v1.actions.plastic import Actions
from agents.dqn_v1.features.plastic_features import PlasticFeatures

# NEW_STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
#                       "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
#                       "BOTTOM LEFT": (0.4, 0.0), "BOTTOM RIGHT": (0.4, 0.0)}
NEW_STARTING_POSITIONS = {
                          "GOAL1": (0.4, 0.0),
                          "GOAL2": (0.4, -0.4),
                          "GOAL3": (0, -0.4),
                          "GOAL4": (0, 0),
                          "FAIL1": (0.4, -0.7),
                          "FAIL2": (0.4, 0.7),
                          #"FAIL2": (-0.5, 0.0)
                        }
STARTING_POSITIONS_NAMES = list(NEW_STARTING_POSITIONS.keys())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--num_episodes', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)

    # Parse Arguments:
    args = parser.parse_args()
    port = args.port
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    
    # Game Interface:
    game_interface = HFOAttackingPlayer(num_opponents=num_op,
                                        num_teammates=num_team, port=port)
    game_interface.connect_to_server()
    # Features Interface:
    features = PlasticFeatures(num_op=num_op, num_team=num_team)
    # Actions Interface:
    actions = Actions()

    features.update_features(game_interface.get_state())

    # print("GO_TO_BALL")
    # for _ in range(50):
    #     hfo_action = (DRIBBLE_TO, -0.5, -0.5)
    #     z = game_interface.step(hfo_action, features.has_ball(), filter=False)
    #     features.update_features(game_interface.get_state())
    
    while not features.has_ball():
        hfo_action = (GO_TO_BALL,)
        z = game_interface.step(hfo_action, features.has_ball(), filter=False)
        features.update_features(game_interface.get_state())
    
    
    print("DRIBBLE")
    for _ in range(50):
        if _ % 2 == 0:
            hfo_action = (DRIBBLE, )
        else:
            hfo_action = (NOOP,)
        z = game_interface.step(hfo_action, features.has_ball(), filter=False)
        features.update_features(game_interface.get_state())

    # print("KICK")
    # hfo_action = (KICK, 80, 0)
    # z = game_interface.step(hfo_action, features.has_ball(), filter=False)
    # features.update_features(game_interface.get_state())

    # print("GO_TO_BALL")
    # while not features.has_ball():
    #     hfo_action = (GO_TO_BALL,)
    #     z = game_interface.step(hfo_action, features.has_ball(), filter=False)
    #     features.update_features(game_interface.get_state())
    
    while True:
        hfo_action = (NOOP,)
        z = game_interface.step(hfo_action, features.has_ball(), filter=False)
        features.update_features(game_interface.get_state())
    
    """
    for ep in range(num_games):
        
        # Update features:
        features.update_features(game_interface.get_state())

        # Starting game position:
        pos_idx = ep % len(NEW_STARTING_POSITIONS)
        start_pos_name = STARTING_POSITIONS_NAMES[pos_idx]
        start_pos_tuple = NEW_STARTING_POSITIONS[start_pos_name]
        print("GOING to {}".format(start_pos_name))
        
        # Start learning loop:
        status = IN_GAME
        prev_action_idx = None
        
        
        while game_interface.in_game():
            actions.dribble_to_pos(start_pos_tuple, features, game_interface)
            if features.has_ball():
                features.update_features(game_interface.get_observation_array())
                actions.shoot_ball(game_interface, features)
            else:
                actions.do_nothing(game_interface, features)

        # Update auxiliar variables:
        if game_interface.scored_goal() or status == GOAL:
            print("[GOAL]")
        else:
            print("[FAIL]")
        # Game Reset
        game_interface.reset()
"""

