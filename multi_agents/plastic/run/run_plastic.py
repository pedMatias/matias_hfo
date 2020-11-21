import argparse
import json
import os
from datetime import datetime as dt

import pandas

from multi_agents.utils import mkdir, export_beliefs_to_graph
from multi_agents.plastic.player import PlasticPlayer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--team_name', type=str, default=None)
    parser.add_argument('--num_teammates', type=int, required=True)
    parser.add_argument('--num_opponents', type=int, required=True)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--memory_bounded', type=str, default="false",
                        choices=("true", "false"))
    parser.add_argument('--history_len', type=int, default=1)
    parser.add_argument('--models_dir', type=str, default=None)
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--save', type=str, choices=("true", "false"),
                        default="true")
    parser.add_argument('--metrics_dir', type=str, default=None)
    parser.add_argument('--prefix', type=str, default="agent")
    args = parser.parse_args()

    team_name = args.team_name
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    memory_bounded = True if args.memory_bounded == "true" else False
    history_len = args.history_len
    directory = args.models_dir
    port = args.port
    save = True if args.save == "true" else False
    metrics_dir = args.metrics_dir
    prefix = args.prefix

    # Start Player:
    player = PlasticPlayer(team_name=team_name, num_teammates=num_team,
                           num_opponents=num_op, port=port,
                           models_dir=directory, memory_bounded=memory_bounded,
                           history_len=history_len)
    
    print(f"[Plastic Player: {team_name}:"
          f"{player.game_interface.hfo.getUnum()}] "
          f"memory_bound={memory_bounded}; history_len={history_len}; "
          f"ep={num_episodes}; num_t={num_team}; num_op={num_op};")
    
    # Explore game:
    selected_teams, game_results, metrics_dict = player.play(
        num_episodes=num_episodes,
        verbose=False)

    # Export train_data:
    if not save:
        print(f"[Not Saving Metrics] Game Metrics: {metrics_dict}")
    else:
        # Metrics dir:
        if not os.path.isdir(metrics_dir):
            raise ValueError(f"metrics_dir is nof found: {metrics_dir}.")
        # Plots
        for part in (50, 20, 10, 1):
            dim = len(selected_teams) // part
            df = pandas.DataFrame(selected_teams[:dim])
            plot_file_name = f"{team_name}_{prefix}_{dim}_selected_teams.png"
            plot_file_name = os.path.join(metrics_dir, plot_file_name)
            export_beliefs_to_graph(df, team_name=team_name,
                                    file_name=plot_file_name)
        
        # Export metrics:
        data = {"team_name": team_name, "number_episodes": num_episodes,
                "memory_bounded?": memory_bounded, "history_len": history_len,
                **metrics_dict}
        metrics_file = f"{team_name}_{prefix}_plastic_metrics.json"
        metrics_file = os.path.join(metrics_dir, metrics_file)
        with open(metrics_file, 'w+') as fp:
            json.dump(data, fp)
        
        # Export game results:
        selected_teams_file = f"{team_name}_{prefix}_selected_teams.json"
        selected_teams_file = os.path.join(metrics_dir, selected_teams_file)
        with open(selected_teams_file, 'w+') as fp:
            json.dump(selected_teams, fp)
            
        game_results_file = f"{team_name}_{prefix}_game_results.json"
        game_results_file = os.path.join(metrics_dir, game_results_file)
        with open(game_results_file, 'w+') as fp:
            json.dump(game_results, fp)

    print(f"\n\n!!!!!!!!! PLASTIC Ended  !!!!!!!!!!!!\n\n")
