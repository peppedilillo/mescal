# fmt: off
import numpy as np


UNBOND = (-1, -1)

dm = {
    'A': ((5, 0), (5, 1), (5, 2), (5, 3), UNBOND, (4, 1), (4, 0), (3, 0),
          (3, 1), (5, 4), (4, 4), (3, 4), (4, 3), (3, 2), (4, 2), (3, 3),
          UNBOND, (2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1),
          (1, 2), (1, 3), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0)),
    'B': (UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND),
    'C': (UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND),
    'D': (UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND)
}

fm1 = {
    'D': ((5, 2), (5, 0), (5, 3), (5, 1), (5, 4), (4, 1), (4, 4), (4, 0),
          (4, 3), (4, 2), (3, 0), (3, 2), UNBOND, (3, 1), (3, 3), (3, 4),
          (2, 4), (2, 3), (2, 1), UNBOND, (2, 2), (2, 0), (1, 2), (1, 3),
          (1, 4), (0, 3), (1, 0), (0, 1), (1, 1), (0, 0), (0, 2), (0, 4)),
    'C': ((5, 4), (5, 0), UNBOND, UNBOND, (5, 3), (5, 2), (5, 1), (4, 1),
          (4, 4), (4, 0), (4, 3), (3, 2), (4, 2), (3, 3), (3, 0), (3, 4),
          (3, 1), (2, 4), (2, 1), (2, 3), (2, 0), (2, 2), (1, 0), (1, 2),
          (1, 3), (1, 4), (1, 1), (0, 4), (0, 1), (0, 3), (0, 0), (0, 2)),
    'B': (UNBOND, (0, 0), (0, 1), (0, 2), (0, 4), (0, 3), (1, 0), (1, 4),
          (1, 3), (1, 1), (2, 4), (1, 2), (2, 3), (2, 2), (2, 0), (2, 1),
          UNBOND, (3, 3), (3, 0), (3, 4), (3, 1), (3, 2), (4, 2), (4, 1),
          (4, 4), (4, 0), (4, 3), (5, 0), (5, 1), (5, 3), (5, 2), (5, 4)),
    'A': ((0, 4), (0, 3), (0, 2), (0, 1), (1, 3), (1, 4), (2, 4), (2, 3),
          (0, 0), (1, 0), (2, 0), (1, 1), (2, 2), (1, 2), (2, 1), (3, 0),
          (3, 1), (3, 2), (3, 3), (3, 4), (4, 4), UNBOND, (4, 3), (4, 2),
          UNBOND, (4, 1), (4, 0), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4))
}

pfm = {
    'D': ((5, 2), (5, 0), (5, 3), (5, 1), (5, 4), (4, 1), (4, 4), (4, 0),
          (4, 3), (4, 2), (3, 0), (3, 2), UNBOND, (3, 1), (3, 3), (3, 4),
          (2, 4), (2, 3), (2, 1), UNBOND, (2, 2), (2, 0), (1, 2), (1, 3),
          (1, 4), (0, 3), (1, 0), (0, 1), (1, 1), (0, 0), (0, 2), (0, 4)),
    'C': ((5, 4), (5, 0), UNBOND, UNBOND, (5, 3), (5, 2), (5, 1), (4, 1),
          (4, 4), (4, 0), (4, 3), (3, 2), (4, 2), (3, 3), (3, 0), (3, 4),
          (3, 1), (2, 4), (2, 1), (2, 3), (2, 0), (2, 2), (1, 0), (1, 2),
          (1, 3), (1, 4), (1, 1), (0, 4), (0, 1), (0, 3), (0, 0), (0, 2)),
    'B': (UNBOND, (0, 0), (0, 1), (0, 2), (0, 4), (0, 3), (1, 0), (1, 4),
          (1, 3), (1, 1), (2, 4), (1, 2), (2, 3), (2, 2), (2, 0), (2, 1),
          UNBOND, (3, 3), (3, 0), (3, 4), (3, 1), (3, 2), (4, 2), (4, 1),
          (4, 4), (4, 0), (4, 3), (5, 0), (5, 1), (5, 3), (5, 2), (5, 4)),
    'A': ((0, 4), (0, 3), (0, 2), (0, 1), (1, 3), (1, 4), (2, 4), (2, 3),
          (0, 0), (1, 0), (2, 0), (1, 1), (2, 2), (1, 2), (2, 1), (3, 0),
          (3, 1), (3, 2), (3, 3), (3, 4), (4, 4), UNBOND, (4, 3), (4, 2),
          UNBOND, (4, 1), (4, 0), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4))
}

fm2 = {
    'D': ((0, 2), UNBOND, (0, 1), (0, 4), (0, 0), (0, 3), (1, 0), (1, 3),
          (1, 1), (1, 4), (2, 4), (1, 2), (2, 3), (2, 2), (2, 0), UNBOND,
          (2, 1), (3, 0), (3, 1), (3, 3), (3, 2), (3, 4), (4, 2), (4, 1),
          (4, 0), (5, 1), (4, 4), (5, 3), (4, 3), (5, 4), (5, 2), (5, 0)),
    'C': (UNBOND, (0, 0), (0, 1), (0, 4), UNBOND, (0, 2), (0, 3), (1, 3),
          (1, 0), (1, 4), (1, 1), (2, 2), (1, 2), (2, 1), (2, 4), (2, 0),
          (2, 3), (3, 0), (3, 3), (3, 1), (3, 4), (3, 2), (4, 4), (4, 2),
          (4, 1), (4, 0), (4, 3), (5, 0), (5, 3), (5, 1), (5, 4), (5, 2)),
    'B': (UNBOND, (5, 4), (5, 3), (5, 2), (5, 0), (5, 1), (4, 4), (4, 0),
          (4, 1), (4, 3), (3, 0), (4, 2), (3, 1), (3, 2), (3, 4), UNBOND,
          (3, 3), (2, 1), (2, 4), (2, 0), (2, 3), (2, 2), (1, 2), (1, 3),
          (1, 0), (1, 4), (1, 1), (0, 4), (0, 3), (0, 1), (0, 2), (0, 0)),
    'A': (UNBOND, (5, 0), (5, 1), (5, 2), (5, 3), (4, 1), (4, 0), (3, 0),
          (3, 1), (5, 4), (4, 4), (3, 4), (4, 3), (3, 2), (4, 2), UNBOND,
          (3, 3), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1),
          (1, 2), (1, 3), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0))
}

fm3 = {
    'D': ((0, 2), (0, 4), (0, 1), (0, 3), (0, 0), UNBOND, (1, 3), (1, 0),
          (1, 4), (1, 1), (1, 2), (2, 4), (2, 2), (2, 3), (2, 1), (2, 0),
          (3, 0), (3, 1), (3, 3), (3, 2), (3, 4), (4, 2), UNBOND, (4, 1),
          (4, 0), (5, 1), (4, 4), (5, 3), (4, 3), (5, 4), (5, 2), (5, 0)),
    'C': ((0, 0), (0, 1), UNBOND, (0, 4), (0, 3), (0, 2), (1, 0), (1, 3),
          (1, 1), (1, 4), (1, 2), (2, 2), (2, 4), (2, 1), (2, 3), (2, 0),
          (3, 0), (3, 3), (3, 1), (3, 4), (3, 2), (4, 4), (4, 2), (4, 1),
          (4, 0), (4, 3), (5, 0), (5, 3), (5, 1), (5, 4), UNBOND, (5, 2)),
    'B': ((5, 4), (5, 3), (5, 2), (5, 0), (5, 1), (4, 4), (4, 0), (4, 1),
          (4, 3), (3, 0), (4, 2), (3, 1), (3, 2), (3, 4), (3, 3), (2, 1),
          (2, 4), (2, 0), (2, 3), (2, 2), (1, 2), (1, 3), UNBOND, (1, 0),
          (1, 4), (1, 1), (0, 4), (0, 3), (0, 1), (0, 2), (0, 0), UNBOND),
    'A': ((5, 0), (5, 1), (5, 2), UNBOND, (5, 3), (4, 1), (4, 0), (3, 0),
          (3, 1), (5, 4), (4, 4), (3, 4), (4, 3), (3, 2), (3, 3), UNBOND,
          (4, 2), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1),
          (1, 2), (1, 3), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0)),
}

fm4 = {
    'D': ((0, 2), (0, 4), (0, 1), UNBOND, (0, 3), (0, 0), UNBOND, (1, 3),
          (1, 0), (1, 4), (1, 1), (1, 2), (2, 4), (2, 2), (2, 3), (2, 1),
          (2, 0), (3, 0), (3, 1), (3, 3), (3, 2), (3, 4), (4, 2), (4, 1),
          (4, 0), (5, 1), (4, 4), (5, 3), (4, 3), (5, 4), (5, 2), (5, 0)),
    'C': ((0, 0), (0, 1), (0, 4), (0, 3), (0, 2), (1, 0), (1, 3), (1, 1),
          (1, 4), (1, 2), (2, 2), (2, 4), (2, 1), (2, 3), (2, 0), (3, 0),
          (3, 3), (3, 1), (3, 4), (3, 2), (4, 4), (4, 2), (4, 1), (4, 0),
          (4, 3), (5, 0), UNBOND, UNBOND, (5, 3), (5, 1), (5, 4), (5, 2)),
    'B': ((5, 4), (5, 3), (5, 2), (5, 0), (5, 1), (4, 4), (4, 0), (4, 1),
          (4, 3), (3, 0), (4, 2), (3, 1), (3, 2), (3, 4), (3, 3), (2, 1),
          (2, 4), (2, 0), (2, 3), (2, 2), UNBOND, (1, 2), UNBOND, (1, 3),
          (1, 0), (1, 4), (1, 1), (0, 4), (0, 3), (0, 1), (0, 2), (0, 0)),
    'A': ((5, 0), (5, 1), (5, 2), UNBOND, (5, 3), (4, 1), (4, 0), (3, 0),
          (3, 1), (5, 4), UNBOND, (3, 4), (4, 4), (3, 2), (4, 3), (3, 3),
          (4, 2), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1),
          (1, 2), (1, 3), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0)),
}


def get_quadrant_map(model: str, quad: str, arr_borders):
    if model in _maps:
        detector_map= _maps[model]
    else:
        raise ValueError("Model Unknown.")

    if quad in ["A", "B", "C", "D"]:
        arr = detector_map[quad]
    else:
        raise ValueError("Unknown quadrant key. Allowed keys are A,B,C,D")

    if arr_borders:
        return tuple(map(lambda x: (x[0] + int(x[0] / 2), x[1]), arr))
    return arr


def get_map(model):
    return {quad: get_quadrant_map(model, quad, arr_borders=False)
            for quad in ['A', 'B', 'C', 'D']}


def get_quad_couples(model, quad):
    qmaparr = np.array(get_quadrant_map(model, quad, arr_borders=True))
    arr = np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]
    dic = dict(arr)
    return dic


def get_couples(model):
    return {q: get_quad_couples(model, q) for q in "ABCD"}


class Detector:
    UNBOND = UNBOND
    def __init__(self, model):
        self.label = model
        self.map = get_map(model)
        self.couples = get_couples(model)


_maps = {
    'dm': dm,
    'fm1': fm1,
    "pfm": pfm,
    "fm2": fm2,
    "fm3": fm3,
    "fm4": fm4,
}

# will run some test on detector maps
if __name__ == '__main__':
    TOT_QUAD = 4
    TOT_CH = 32
    TOT_BONDED = 30
    ROWS = 6
    COLS = 5

    for model, map in _maps.items():
        # checks for 4 quadrants
        assert len(map.keys()) == TOT_QUAD
        for quadrant in map.keys():
            print("testing quadrant {} of {}..".format(quadrant, model))
            channels = [sdd for sdd in map[quadrant]]
            bonded = [sdd for sdd in map[quadrant] if sdd != UNBOND]
            # checks for 32 channels
            assert len(channels) == TOT_CH
            # checks for no duplicates in each quadrant
            assert len(bonded) == len(set(bonded))
            # check no entry out of grid
            for row, col in bonded:
                assert row < ROWS
                assert col < COLS
            if not ((model=='dm') and quadrant in ['B', 'C', 'D']):
                # checks for 2 unbounded channels
                assert len(set(bonded)) == TOT_BONDED
                # check for all places to be assigned
                for row in range(ROWS):
                    for column in range(COLS):
                        assert (row, column) in bonded
    print("Good news! All tests passed.")
