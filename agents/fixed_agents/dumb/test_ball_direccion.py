import argparse

from hfo import IN_GAME, GOAL

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from agents.dqn_v1.actions.simple import Actions
from agents.dqn_v1.features.discrete_features import DiscFeatures1Teammate

STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                      "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                      "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--num_games', type=int, default=1)

    args = parser.parse_args()
    port = args.port
    num_games = args.num_games
    
    NUM_TEAMMATES = 0
    NUM_OPPONENTS = 1
    
    # Game Interface:
    game_interface = HFOAttackingPlayer(num_opponents=NUM_OPPONENTS,
                                        num_teammates=NUM_TEAMMATES, port=port)
    game_interface.connect_to_server()
    # Features Interface:
    features = DiscFeatures1Teammate(num_op=NUM_OPPONENTS,
                                     num_team=NUM_TEAMMATES)
    # Actions Interface:
    actions = Actions()

    for ep in range(num_games):
        
        # Update features:
        features.update_features(game_interface.get_state())
        # Set up gaming conditions:
        actions.dribble_to_pos((-0.5, -0.7), features, game_interface)
        
        # Start learning loop
        status = IN_GAME
        prev_action_idx = None
        while game_interface.in_game():
            if features.has_ball():
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

