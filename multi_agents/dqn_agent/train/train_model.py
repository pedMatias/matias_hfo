#!/usr/bin/hfo_env python3
# encoding utf-8
import argparse
import json
import os

from multi_agents.dqn_agent.trainer import Trainer, TrainMetrics

"""
This module is used to train the model using exploration data
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_teammates', type=int, default=0, required=True)
    parser.add_argument('--num_opponents', type=int, default=0, required=True)
    parser.add_argument('--team_name', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--dir', type=str, required=True)
    
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    team_name = args.team_name
    step = args.step
    directory = args.dir
    
    # Start Trainer:
    trainer = Trainer(num_teammates=num_team, num_opponents=num_op, step=step,
                      directory=directory)
    trainer.load_experience_from_dir(clean_learn_buffer=True, verbose=True)
    metrics: TrainMetrics = trainer.train_model(verbose=True)

    metrics_file = f"train_metrics.{step}.json"
    metrics_file = os.path.join(directory, metrics_file)
    with open(metrics_file, 'w+') as fp:
        json.dump(metrics._asdict(), fp)
    
    print("\n!!!!!!!!! Train End !!!!!!!!!!!!\n\n")
