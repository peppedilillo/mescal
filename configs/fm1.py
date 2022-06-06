import numpy as np

UNBOND = (-1, -1)

MAP_QUAD_D = ((5, 2), (5, 0), (5, 3), (5, 1), (5, 4), (4, 1), (4, 4), (4, 0),
              (4, 3), (4, 2), (3, 0), (3, 2), UNBOND, (3, 1), (3, 3), (3, 4),
              (2, 4), (2, 3), (2, 1), UNBOND, (2, 2), (2, 0), (1, 2), (1, 3),
              (1, 4), (0, 3), (1, 0), (0, 1), (1, 1), (0, 0), (0, 2), (0, 4))
MAP_QUAD_C = ((5, 4), (5, 0), UNBOND, UNBOND, (5, 3), (5, 2), (5, 1), (4, 1),
              (4, 4), (4, 0), (4, 3), (3, 2), (4, 2), (3, 3), (3, 0), (3, 4),
              (3, 1), (2, 4), (2, 1), (2, 3), (2, 0), (2, 2), (1, 0), (1, 2),
              (1, 3), (1, 4), (1, 1), (0, 4), (0, 1), (0, 3), (0, 0), (0, 2))
MAP_QUAD_B = (UNBOND, (0, 0), (0, 1), (0, 2), (0, 4), (0, 3), (1, 0), (1, 4),
              (1, 3), (1, 1), (2, 4), (1, 2), (2, 3), (2, 2), (2, 0), (2, 1),
              UNBOND, (3, 3), (3, 0), (3, 4), (3, 1), (3, 2), (4, 2), (4, 1),
              (4, 4), (4, 0), (4, 3), (5, 0), (5, 1), (5, 3), (5, 2), (5, 4))
MAP_QUAD_A = ((0, 4), (0, 3), (0, 2), (0, 1), (1, 3), (1, 4), (2, 4), (2, 3),
              (0, 0), (1, 0), (2, 0), (1, 1), (2, 2), (1, 2), (2, 1), (3, 0),
              (3, 1), (3, 2), (3, 3), (3, 4), (4, 4), UNBOND, (4, 3), (4, 2),
              UNBOND, (4, 1), (4, 0), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4))


def get_qmap(quad: str, arr_borders=True):
    if quad == 'D':
        arr = MAP_QUAD_D
    elif quad == 'C':
        arr = MAP_QUAD_C
    elif quad == 'B':
        arr = MAP_QUAD_B
    elif quad == 'A':
        arr = MAP_QUAD_A
    else:
        raise ValueError("Unknown quadrant key. Allowed keys are A,B,C,D")

    if arr_borders:
        return tuple(map(lambda x: (x[0] + int(x[0] / 2), x[1]), arr))
    return arr


def get_channels(quad: str):
    return [ch for ch, _ in enumerate(get_qmap(quad)) if ch is not UNBOND]


def get_couples(quad: str):
    qmaparr = np.array(get_qmap(quad))
    return np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]
