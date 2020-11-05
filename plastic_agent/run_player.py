import argparse
import json
import os
import pickle

from plastic_agent import config
from plastic_agent.player.player import Player

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=("exploration", "testing"),
                        required=True)
    parser.add_argument('--team_name', type=str, default=None)
    parser.add_argument('--num_teammates', type=int, required=True)
    parser.add_argument('--num_opponents', type=int, required=True)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--test_id', type=int, default=0)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--no_save', type=str,
                        choices=("true", "false"), default="false")
    args = parser.parse_args()
    
    mode = args.mode
    team_name = args.team_name
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    test_id = args.test_id
    step = args.step
    directory = args.dir
    port = args.port
    model_file = args.model_file
    epsilon = args.epsilon
    no_save = True if args.no_save == "true" else False
    print(f"[{mode.upper()}:{step}] ep={num_episodes}; "
          f"num_t={num_team}; num_op={num_op};")

    # Start Player:
    player = Player(team_name=team_name, num_teammates=num_team,
                    num_opponents=num_op, port=port, epsilon=epsilon,
                    model_file=model_file)
    # Explore game:
    learn_buffer, metrics_dict = player.play(num_episodes=num_episodes)

    # Export train_data:
    if no_save:
        print(f"[Not Saving Metrics] Game Metrics: {metrics_dict}")
    elif mode == "exploration":
        learn_buffer_file = config.EXPERIENCE_BUFFER_FORMAT.format(step=step)
        train_data_file = os.path.join(directory, learn_buffer_file)
        with open(train_data_file, "wb") as fp:
            pickle.dump(learn_buffer, fp)
        # Export metrics:
        data = {"number_episodes": num_episodes, **metrics_dict}
        metrics_file = f"{mode}_metrics.{step}.json"
        metrics_file = os.path.join(directory, metrics_file)
        with open(metrics_file, 'w+') as fp:
            json.dump(data, fp)
    # Test Mode:
    else:
        metrics_file = f"{mode}_metrics.{step}.json"
        metrics_file = os.path.join(directory, metrics_file)
        data = {"number_episodes": num_episodes, **metrics_dict}
        if test_id > 0:
            with open(metrics_file, "rb") as fp:
                prev_data = json.load(fp)
            prev_data[test_id] = data
        else:
            prev_data = {test_id: data}
        with open(metrics_file, 'w+') as fp:
            json.dump(prev_data, fp)

    print(f"\n\n!!!!!!!!! {mode.upper()} Ended  !!!!!!!!!!!!\n\n")
