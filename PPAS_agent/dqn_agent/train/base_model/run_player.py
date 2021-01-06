import argparse
import json
import os
import pickle

from multi_agents import config
from multi_agents.dqn_agent.train.base_model.fixed_player import FixedPlayer

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
    parser.add_argument('--pass_ball', type=str,
                        choices=("true", "false"), default="false")
    parser.add_argument('--learning_boost', type=str,
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
    pass_ball = True if args.pass_ball == "true" else False
    learning_boost = True if args.learning_boost == "true" else False
    print(f"[{mode.upper()}:{step}] ep={num_episodes}; "
          f"num_t={num_team}; num_op={num_op}; "
          f"learning_boost={learning_boost}")

    # Start Player:
    player = FixedPlayer(team_name=team_name, num_teammates=num_team,
                         num_opponents=num_op, port=port, epsilon=epsilon,
                         model_file=model_file, learning_boost=learning_boost,
                         pass_ball=pass_ball)
    # Explore game:
    learn_buffer, metrics_dict = player.play(num_episodes=num_episodes,
                                             verbose=False)

    # Export train_data:
    if no_save:
        print(f"[Not Saving Metrics] Game Metrics: {metrics_dict}")
    elif mode == "exploration":
        # DQN Experience:
        learn_buffer_file = config.DQN_EXPERIENCE_BUFFER_FORMAT.format(step=step)
        train_data_file = os.path.join(directory, learn_buffer_file)
        with open(train_data_file, "wb") as fp:
            pickle.dump(learn_buffer.dqn_buffer, fp)
        
        # Team experience buffer:
        learn_buffer_file = config.TEAM_EXPERIENCE_BUFFER_FORMAT.format(
            step=step)
        train_data_file = os.path.join(directory, learn_buffer_file)
        with open(train_data_file, "wb") as fp:
            pickle.dump(learn_buffer.team_experience_buffer, fp)
        
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
        if test_id > 0 and os.path.isfile(metrics_file):
            with open(metrics_file, "rb") as fp:
                prev_data = json.load(fp)
            prev_data[test_id] = data
        else:
            prev_data = {test_id: data}
        with open(metrics_file, 'w+') as fp:
            json.dump(prev_data, fp)

    print(f"\n\n!!!!!!!!! {mode.upper()} Ended  !!!!!!!!!!!!\n\n")
