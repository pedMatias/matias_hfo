#!/usr/bin/env python3
# encoding utf-8
from datetime import date, datetime as dt
import os

import numpy as np

from matias_hfo import settings
from matias_hfo.agents.utils import ServerDownError, NoActionPlayedError
        

def mkdir(name: str, idx: int = None, **kwargs):
    today = date.today()
    name_dir = ""
    # Train type:
    name_dir += str(name) + "_"
    # Extra arguments:
    for key, value in kwargs.items():
        name_dir += str(value) + str(key) + "_"
    # Date:
    name_dir += today.isoformat()
    
    if isinstance(idx, int):
        name_dir += "_" + str(idx)
    
    path = os.path.join(settings.MODELS_DIR, name_dir)
    try:
        os.mkdir(path)
    except FileExistsError:
        if idx is None:
            idx = 1
        else:
            idx += 1
        mkdir(name, idx=idx, **kwargs)
    return path


def save_model(q_table: str, directory: str, file_name: str):
    file_path = os.path.join(directory, file_name)
    np.save(file_path, q_table)


def print_transiction(arr: tuple, actions_instance):
    """ Transcition array format
    (observation space, action, reward, new observation space, done) """
    def round_list(lista):
        return [round(el, 2) for el in lista]
    print("+ D?:{}; {} -> {}; R:{}; Act:{};".format(
        arr[4],
        round_list(arr[0].tolist()),
        round_list(arr[3].tolist()),
        arr[2],
        actions_instance.actions[arr[1]]
    ))