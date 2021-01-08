import argparse
import json
import os
from datetime import datetime as dt

import pandas

from multi_agents.utils import mkdir, export_beliefs_to_graph
from multi_agents.plastic.plastic_client.player import PlasticClientPlayer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Team:
    parser.add_argument('--team_name', type=str, default=None)
    parser.add_argument('--num_teammates', type=int, required=True)
    parser.add_argument('--num_opponents', type=int, required=True)
    parser.add_argument('--num_episodes', type=int, default=1)
    # Directories:
    parser.add_argument('--metrics_dir', type=str, default=None)
    parser.add_argument('--models_dir', type=str, default=None)
    # Plastic Set-up:
    parser.add_argument('--model_type', type=str, default="stochastic",
                        choices=("stochastic", "adversarial"), required=True)
    parser.add_argument('--agent_type', type=str, default="plastic",
                        choices=("plastic", "memory_bounded", "correct_policy",
                                 "random"),
                        required=True)
    parser.add_argument('--history_len', type=int, default=1)
    # Metrics:
    parser.add_argument('--plot', type=str, choices=("true", "false"),
                        default="true")
    parser.add_argument('--save', type=str, choices=("true", "false"),
                        default="true")
    # Others:
    parser.add_argument('--prefix', type=str, default="agent")
    parser.add_argument('--use_webservice', type=str, choices=("true", "false"),
                        default="false")
    parser.add_argument('--port', type=int, default=6000)
    
    args = parser.parse_args()

    # Team:
    team_name = args.team_name
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    # Directories:
    directory = args.models_dir
    metrics_dir = args.metrics_dir
    # Plastic Set-up:
    agent_type = args.agent_type
    history_len = args.history_len
    model_type = args.model_type
    # Metrics:
    save = True if args.save == "true" else False
    plot_metrics = True if args.plot == "true" else False
    # Others:
    prefix = args.prefix
    use_webservice = True if args.use_webservice == "true" else False
    port = args.port

    # Start Player:
    player = PlasticClientPlayer(team_name=team_name, num_teammates=num_team,
                                 num_opponents=num_op, port=port,
                                 models_dir=directory,
                                 agent_type=agent_type,
                                 history_len=history_len,
                                 model_type=model_type,
                                 use_webservice=use_webservice)
    
    print(f"[Plastic Player: {team_name}:"
          f"{player.game_interface.hfo.getUnum()}: {agent_type}] "
          f"model_type={model_type};  history_len={history_len}; "
          f"ep={num_episodes}; num_t={num_team}; num_op={num_op}; "
          f"plot_metrics? {plot_metrics};")
    
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
        if plot_metrics:
            # Plots
            for part in (50, 20, 10, 1):
                dim = len(selected_teams) // part
                df = pandas.DataFrame(selected_teams[:dim])
                plot_file_name = f"{team_name}_{prefix}_{dim}_" \
                                 f"selected_teams.png"
                plot_file_name = os.path.join(metrics_dir, plot_file_name)
                export_beliefs_to_graph(df, team_name=team_name,
                                        file_name=plot_file_name)
        
        # Export metrics:
        data = {"team_name": team_name, "number_episodes": num_episodes,
                "agent_type?": agent_type, "history_len": history_len,
                "model_type": model_type, **metrics_dict}
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
