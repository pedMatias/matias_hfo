# !/usr/bin/hfo_env python3
# encoding utf-8
import argparse
import os
import pickle

from agents.plastic_dqn_v1.plastic.policy import Policy
from agents.plastic_dqn_v1 import config
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--team_name', type=str, default=None)
    parser.add_argument('--dir', type=str)
    
    # Parse arguments:
    args = parser.parse_args()
    team_name = args.team_name
    directory = args.dir
    
    print(f"[PLASTIC Train: {team_name}] dir={directory};")

    # Model file:
    policy = Policy.create(team_name, dir=directory)
    policy.save_plastic_model(dir=directory)
    
    print("\n!!!!!!!!! Train End !!!!!!!!!!!!\n\n")