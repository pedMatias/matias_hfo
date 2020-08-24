import random

import numpy as np


class NoActionPlayedError(Exception):
    pass


class ServerDownError(Exception):
    pass


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
