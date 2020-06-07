#!/usr/bin/env python3
# encoding utf-8
from datetime import datetime as dt
import os

import numpy as np

from matias_hfo import settings
from matias_hfo.agents.utils import ServerDownError, NoActionPlayedError
        

def mkdir(name=None, num_episodes=None, num_op=None, extra_note=None):
    now = dt.now().replace(second=0, microsecond=0)
    name_dir = ""
    if extra_note:
        name_dir += str(extra_note) + "_"
    if num_episodes:
        name_dir += str(num_episodes) + "ep_"
    if num_op:
        name_dir += str(num_op) + "op_"
    name_dir += now.strftime("%Y-%m-%d_%H:%M:%S")
    path = os.path.join(settings.MODELS_DIR, name_dir)
    try:
        os.mkdir(path)
    except FileExistsError:
        name_dir += "_2"
        path = os.path.join(settings.MODELS_DIR, name_dir)
        os.mkdir(path)
    return path


def save_model(q_table: str, directory: str, file_name: str):
    file_path = os.path.join(directory, file_name)
    np.save(file_path, q_table)
