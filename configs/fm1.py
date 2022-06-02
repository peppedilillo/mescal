import numpy as np

UNBONDED = (-1, -1)
MAP_QUAD_D = ((0, 2), (0, 4), (0, 1), (0, 3), (0, 0), (1, 3), (1, 0), (1, 4),
              (1, 1), (1, 2), (2, 4), (2, 2), UNBONDED, (2, 3), (2, 1), (2, 0),
              (3, 0), (3, 1), (3, 3), UNBONDED, (3, 2), (3, 4), (4, 2), (4, 1),
              (4, 0), (5, 1), (4, 4), (5, 3), (4, 3), (5, 4), (5, 2), (5, 0))


def get_qmap(quad: str, arr_borders=True):
    if quad == 'D':
        arr = MAP_QUAD_D
    else:
        raise ValueError("not yet implemented")
    if arr_borders:
        return tuple(map(lambda x: (x[0] + int(x[0] / 2), x[1]), arr))
    return arr


def get_channels(quad: str):
    return [ch for ch, _ in enumerate(get_qmap(quad)) if ch is not UNBONDED]


def get_couples(quad: str):
    qmaparr = np.array(get_qmap(quad))
    return np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]
