#!/usr/bin/hfo_env python3
# encoding utf-8
import argparse
import json
import os
import pickle


from agents.plastic_dqn_v1.hfo_env.player import Player
from agents.plastic_dqn_v1 import config


"""
This module is used to gather data using previous trained deep q neural nets,
with specific epsilons
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--team_name', type=str, default=None)
    parser.add_argument('--starts_fixed_position', type=str, default="false")
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--port', type=int, default=6000)
    
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    team_name = args.team_name
    starts_fixed_position = True if args.starts_fixed_position == "true" \
        else False
    step = args.step
    epsilon = args.epsilon
    directory = args.dir
    port = args.port
    
    print(f"[Explore Player: {team_name}-{step}] ep={num_episodes}; "
          f"num_t={num_team}; num_op={num_op}; epsilon={epsilon}; "
          f"starts_fixed_pos={starts_fixed_position};")
    
    if step == 0:
        # Load base model:
        model_file = config.BASE_MODEL_DQN
    else:
        # Beginning of a stage:
        prev_step = step - 1
        # Get model file:
        model_file = os.path.join(directory, f"{team_name}_{prev_step}.model")
        print("MODEL FILE: ", model_file)

    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op,
                    port=port, epsilon=epsilon, model_file=model_file)
    
    # Explore game:
    learn_buffer, metrics_dict = player.play(
            num_episodes=num_episodes,
            starts_fixed_position=starts_fixed_position)
    
    # Export train_data:
    learn_buffer_file = config.EXPERIENCE_BUFFER_FORMAT.format(
        team_name=team_name, step=step)
    train_data_file = os.path.join(directory, learn_buffer_file)
    with open(train_data_file, "wb") as fp:
        pickle.dump(learn_buffer, fp)
    
    # Export metrics:
    data = {"number_episodes": num_episodes,
            "starts_fixed_position": starts_fixed_position,
            **metrics_dict}
    
    metrics_file = os.path.join(directory,
                                f"{team_name}_exploration_metrics_{step}.json")
    with open(metrics_file, 'w+') as fp:
        json.dump(data, fp)
        
    print("\n\n!!!!!!!!! Exploration Ended  !!!!!!!!!!!!\n\n")
