# !/usr/bin/hfo_env python3
# encoding utf-8
import argparse
import os
import pickle

import numpy as np

from agents.plastic_dqn_v1.agent.replay_buffer import ExperienceBuffer, \
    LearnBuffer
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
    
    step = 0
    experience_file = config.EXPERIENCE_BUFFER_FORMAT.format(step=step)
    data_file = os.path.join(directory, team_name, experience_file)
    print("File ", data_file)
    replay_buffer = list()
    while os.path.isfile(data_file):
        with open(data_file, "rb") as fp:
            learn_buffer: LearnBuffer = pickle.load(fp)
            data: list = learn_buffer.buffer
            replay_buffer += data
            print(f"Add stage {step} data. SIZE={len(data)}")
        step += 1
        experience_file = config.EXPERIENCE_BUFFER_FORMAT.format(
            team_name=team_name, step=step)
        data_file = os.path.join(directory, team_name, experience_file)
        print("File ", data_file)

    experience_buffer = ExperienceBuffer(np.array(replay_buffer))
    experience_buffer.save_to_pickle(dir=directory, team_name=team_name)
    
    print("\n!!!!!!!!! Train End !!!!!!!!!!!!\n\n")