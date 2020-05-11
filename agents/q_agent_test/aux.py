from datetime import datetime as dt
import os

import numpy as np

import settings


def mkdir():
    now = dt.now().replace(second=0, microsecond=0)
    name_dir = "q_agent_train_" + now.strftime("%Y-%m-%d_%H:%M:%S")
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
