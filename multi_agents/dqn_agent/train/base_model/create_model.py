#!/usr/bin/hfo_env python3
# encoding utf-8
import argparse

from multi_agents.dqn_agent.trainer import Trainer

"""
This module is used to train the model using exploration data
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_teammates', type=int, default=0, required=True)
    parser.add_argument('--num_opponents', type=int, default=0, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    model_name = args.model_name
    directory = args.dir
    step = 0
    
    # Start Trainer:
    trainer = Trainer(num_teammates=num_team, num_opponents=num_op, step=step,
                      directory=directory)
    trainer.dqn.save_model(model_name)
