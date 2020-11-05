import argparse
import json
import os
from datetime import datetime as dt

import pandas

from plastic_agent.utils import mkdir, export_beliefs_to_graph
from plastic_agent.plastic.player import PlasticPlayer

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
    print(f"[Plastic Player: {team_name}] memory_bound={memory_bounded}; "
          f"history_len={history_len}; ep={num_episodes}; num_t={num_team};"
          f" num_op={num_op};")

    # Start Player:
    player = PlasticPlayer(team_name=team_name, num_teammates=num_team,
                           num_opponents=num_op, port=port,
                           models_dir=directory, memory_bounded=memory_bounded,
                           history_len=history_len)
    # Explore game:
    selected_teams, metrics_dict = player.play(num_episodes=num_episodes,
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
            plot_file_name = f"{team_name}_{dim}_selected_teams.png"
            plot_file_name = os.path.join(metrics_dir, plot_file_name)
            export_beliefs_to_graph(df, team_name=team_name,
                                    file_name=plot_file_name)
        
        # Export metrics:
        data = {"team_name": team_name, "number_episodes": num_episodes,
                "memory_bounded?": memory_bounded, "history_len": history_len,
                **metrics_dict}
        metrics_file = f"{team_name}_plastic_metrics.json"
        metrics_file = os.path.join(metrics_dir, metrics_file)
        with open(metrics_file, 'w+') as fp:
            json.dump(data, fp)

    print(f"\n\n!!!!!!!!! PLASTIC Ended  !!!!!!!!!!!!\n\n")
