from datetime import date, datetime as dt
import os
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plastic_agent import config
from matias_hfo import settings


class NoActionPlayedError(Exception):
    pass


class ServerDownError(Exception):
    pass


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


def print_transiction(arr: tuple, actions_instance, simplex=False):
    """ Transcition array format
    (observation space, action, reward, new observation space, done) """
    
    def round_list(lista):
        return [round(el, 2) for el in lista]
    
    if simplex:
        print("+ R:{}; Act:{} D?:{}; {};".format(
            arr[2],
            actions_instance.actions[arr[1]],
            arr[4],
            round_list(arr[0].tolist())
        ))
    else:
        print("+ D?:{}; {} -> {}; R:{}; Act:{};".format(
            arr[4],
            round_list(arr[0].tolist()),
            round_list(arr[3].tolist()),
            arr[2],
            actions_instance.actions[arr[1]]
        ))


def check_same_model(model1, model2):
    obs1 = np.array([-0.38, -0.0, -0.16, -0.89, -0.11, -1.0, -0.92, 0.2, 0.0])
    obs2 = np.array(
        [-0.28, 0.34, -0.03, -0.88, -0.13, -1.0, -0.86, -0.12, 0.0])
    obs3 = np.array([0.49, 0.16, -0.23, -0.65, -0.72, 1.0, -0.61, -0.62, 0.0])
    for obs in [obs1, obs2, obs3]:
        state = obs[np.newaxis, :]
        qs1 = model1.predict(state)
        qs2 = model2.predict(state)
        if qs1.all() != qs2.all():
            print("Different Models: ", qs1, qs2)
            return False
    return True


def get_angle(goalie: np.ndarray, player: np.ndarray, point: np.ndarray):
    """
    Returns
    -------
    y : ndarray of floats
        The corresponding degree values; if `out` was supplied this is a
        reference to it.
        This is a scalar if `x` is a scalar.
    """
    # Check for invalid values:
    for array in [goalie, player, point]:
        if array[0] == -2 or array[1] == -2:
            return 0
    
    a = np.array(goalie)
    b = np.array(player)
    c = np.array(point)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def get_opposite_vector(pos: np.ndarray, t_pos: np.ndarray) -> np.ndarray:
    vect = pos - t_pos
    if np.linalg.norm(vect) <= 0.05:
        val1 = vect[0] + (random.randrange(-9, 9) / 100)
        val2 = vect[1] + (random.randrange(-9, 9) / 100)
        vect = np.array([val1, val2])
        return vect
    else:
        if abs(vect[0]) > 0.9:
            vect[0] = 0.9 if vect[0] > 0 else -0.9
        if abs(vect[1]) > 0.9:
            vect[1] = 0.9 if vect[1] > 0 else -0.9
        return vect


def get_vertices_around_ball(ball_pos: list) -> list:
    """Offensive players start on randomly selected vertices forming a square
    around the ball with edge length 0.2 length with an added offset uniformly
    randomly selected in[0, 0.1 length] """
    vertices = []
    for x in [ball_pos[0] - 0.1, ball_pos[0] + 0.1]:
        x += random.randrange(-10, 10) / 100
        for y in [ball_pos[1] - 0.1, ball_pos[1] + 0.1]:
            y += random.randrange(-10, 10) / 100
            if -0.9 < x < 0.9 and -0.9 < y < 0.9:
                vertices.append([x, y])
    if len(vertices) == 0:
        vertices.append([-0.5, 0])
    return vertices


def export_beliefs_to_graph(df: pd.DataFrame, team_name: str, file_name: str):
    fig = go.Figure()
    fig.update_layout(title_text=f"Play with team {team_name}")
    fig.update_xaxes(title_text="games")
    for t in config.TEAMS_NAMES:
        fig.add_trace(go.Scatter(x=df.index, y=df[t], mode='lines', name=t))
    fig.write_image(file_name)
