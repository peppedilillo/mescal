import numpy as np

UNBOND = (-1, -1)

# fmt: off
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
    'D': ((0, 2), (0, 4), (0, 1), (0, 3), (0, 0), (1, 3), UNBOND, (1, 0),
          (1, 4), (1, 1), (1, 2), UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND,
          UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND, UNBOND)
}

fm1 = {
    'D': ((0, 2), (0, 4), (0, 1), (0, 3), (0, 0), (1, 3), (1, 0), (1, 4),
          (1, 1), (1, 2), (2, 4), (2, 2), UNBOND, (2, 3), (2, 1), (2, 0),
          (3, 0), (3, 1), (3, 3), UNBOND, (3, 2), (3, 4), (4, 2), (4, 1),
          (4, 0), (5, 1), (4, 4), (5, 3), (4, 3), (5, 4), (5, 2), (5, 0)),
    'C': ((0, 0), (0, 4), UNBOND, UNBOND, (0, 1), (0, 2), (0, 3), (1, 3),
          (1, 0), (1, 4), (1, 1), (2, 2), (1, 2), (2, 1), (2, 4), (2, 0),
          (2, 3), (3, 0), (3, 3), (3, 1), (3, 4), (3, 2), (4, 4), (4, 2),
          (4, 1), (4, 0), (4, 3), (5, 0), (5, 3), (5, 1), (5, 4), (5, 2)),
    'B': (UNBOND, (5, 4), (5, 3), (5, 2), (5, 0), (5, 1), (4, 4), (4, 0),
          (4, 1), (4, 3), (3, 0), (4, 2), (3, 1), (3, 2), (3, 4), (3, 3),
          UNBOND, (2, 1), (2, 4), (2, 0), (2, 3), (2, 2), (1, 2), (1, 3),
          (1, 0), (1, 4), (1, 1), (0, 4), (0, 3), (0, 1), (0, 2), (0, 0)),
    'A': ((5, 0), (5, 1), (5, 2), (5, 3), (4, 1), (4, 0), (3, 0), (3, 1),
          (5, 4), (4, 4), (3, 4), (4, 3), (3, 2), (4, 2), (3, 3), (2, 4),
          (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), UNBOND, (1, 1), (1, 2),
          UNBOND, (1, 3), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0))
}

pfm = {
    'D': ((0, 2), (0, 4), (0, 1), (0, 3), (0, 0), (1, 3), (1, 0), (1, 4),
          (1, 1), (1, 2), (2, 4), (2, 2), UNBOND, (2, 3), (2, 1), (2, 0),
          (3, 0), (3, 1), (3, 3), UNBOND, (3, 2), (3, 4), (4, 2), (4, 1),
          (4, 0), (5, 1), (4, 4), (5, 3), (4, 3), (5, 4), (5, 2), (5, 0)),
    'C': ((0, 0), (0, 4), UNBOND, UNBOND, (0, 1), (0, 2), (0, 3), (1, 3),
          (1, 0), (1, 4), (1, 1), (2, 2), (1, 2), (2, 1), (2, 4), (2, 0),
          (2, 3), (3, 0), (3, 3), (3, 1), (3, 4), (3, 2), (4, 4), (4, 2),
          (4, 1), (4, 0), (4, 3), (5, 0), (5, 3), (5, 1), (5, 4), (5, 2)),
    'B': (UNBOND, (5, 4), (5, 3), (5, 2), (5, 0), (5, 1), (4, 4), (4, 0),
          (4, 1), (4, 3), (3, 0), (4, 2), (3, 1), (3, 2), (3, 4), (3, 3),
          UNBOND, (2, 1), (2, 4), (2, 0), (2, 3), (2, 2), (1, 2), (1, 3),
          (1, 0), (1, 4), (1, 1), (0, 4), (0, 3), (0, 1), (0, 2), (0, 0)),
    'A': ((5, 0), (5, 1), (5, 2), (5, 3), (4, 1), (4, 0), (3, 0), (3, 1),
          (5, 4), (4, 4), (3, 4), (4, 3), (3, 2), (4, 2), (3, 3), (2, 4),
          (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), UNBOND, (1, 1), (1, 2),
          UNBOND, (1, 3), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0))
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

fm5 = {
    'D': ((0, 2), (0, 4), (0, 1), (0, 3), (0, 0), (1, 3), (1, 0), (1, 4),
          (1, 1), (1, 2), UNBOND, (2, 4), (2, 2), (2, 3), (2, 1), (2, 0),
          (3, 0), UNBOND, (3, 1), (3, 3), (3, 2), (3, 4), (4, 2), (4, 1),
          (4, 0), (5, 1), (4, 4), (5, 3), (4, 3), (5, 4), (5, 2), (5, 0)),
    'C': ((0, 0), (0, 1), (0, 4), (0, 3), (0, 2), UNBOND, (1, 0), (1, 3),
          (1, 1), (1, 4), (1, 2), (2, 2), (2, 4), UNBOND, (2, 1), (2, 3),
          (2, 0), (3, 0), (3, 3), (3, 1), (3, 4), (3, 2), (4, 4), (4, 2),
          (4, 1), (4, 0), (4, 3), (5, 0), (5, 3), (5, 1), (5, 4), (5, 2)),
    'B': ((5, 4), (5, 3), (5, 2), (5, 0), (5, 1), (4, 4), (4, 0), (4, 1),
          (4, 3), (3, 0), (4, 2), (3, 1), (3, 2), (3, 4), (3, 3), (2, 1),
          (2, 4), UNBOND, (2, 0), (2, 3), (2, 2), (1, 2), (1, 3), (1, 0),
          (1, 4), (1, 1), UNBOND, (0, 4), (0, 3), (0, 1), (0, 2), (0, 0)),
    'A': ((5, 0), (5, 1), (5, 2), (5, 3), (4, 1), (4, 0), UNBOND, (3, 0),
          (3, 1), (5, 4), (4, 4), (3, 4), (4, 3), (3, 2), (4, 2), (3, 3),
          (2, 4), UNBOND, (2, 3), (2, 1), (2, 2), (1, 0), (2, 0), (1, 2),
          (1, 1), (1, 3), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0)),
}

fm6 = {
    'D': ((0, 2), (0, 4), (0, 1), (0, 3), (0, 0), (1, 3), (1, 0), (1 ,4),
          (1, 1), UNBOND, (1, 2), (2, 4), (2, 2), (2 ,3), (2, 1), (2, 0),
          (3, 0), (3, 1), (3, 3), (3, 2), (3, 4), UNBOND, (4, 2), (4, 1),
          (4, 0), (5, 1), (4, 4), (5, 3), (4, 3), (5, 4), (5, 2), (5, 0),),
    'C': ((0, 0), (0, 1), (0 ,3), (0, 4), (0, 2), UNBOND, (1, 0), (1, 3),
          (1, 1), (1 ,4), (1 ,2), (2, 2), (2, 4), (2, 1), (2, 3), (2 ,0),
          (3, 0), (3, 3), UNBOND, (3, 1), (3, 4), (3, 2), (4, 4), (4, 2),
          (4, 1), (4, 0), (4, 3), (5, 0), (5, 3), (5, 1), (5, 4), (5, 2)),
    'B': ((5, 4), (5, 3), (5, 2), UNBOND, (5, 0), (5, 1), (4, 4), (4, 0),
          (4, 1), (4, 3), (3, 0), (4, 2), (3, 1), (3, 2), (3, 4), (3, 3),
          (2, 1), UNBOND, (2, 4), (2, 0), (2, 3), (2, 2), (1, 2), (1, 3),
          (1, 0), (1, 4), (1, 1), (0, 4), (0, 3), (0, 1), (0, 2), (0, 0)),
    'A': ((5, 0), (5, 1), (5, 2), (5, 3), UNBOND, (4, 1), (3, 0), (4, 0),
          (3, 1), (5, 4), (4, 4), (3, 4), (4, 3), (3, 2), (4, 2), (3, 3),
          UNBOND, (2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1),
          (1, 2), (1, 3), (1, 4), (0, 4), (0, 3), (0, 1), (0, 2), (0, 0)),
}
# fmt: on


_maps = {
    "dm": dm,
    "pfm": pfm,
    "fm1": fm1,
    "fm2": fm2,
    "fm3": fm3,
    "fm4": fm4,
    "fm5": fm5,
    "fm6": fm6,
}


def supported_models():
    return list(_maps.keys())


def _get_quadrant_map(model: str, quad: str, arr_borders):
    detector_map = _maps[model]
    arr = detector_map[quad]
    if arr_borders:
        return tuple(map(lambda x: (x[0] + int(x[0] / 2), x[1]), arr))
    return arr


def get_map(model):
    return {
        quad: _get_quadrant_map(model, quad, arr_borders=False)
        for quad in ["A", "B", "C", "D"]
    }


def get_couples(model):
    couples_dict = {}
    for quad in "ABCD":
        qmaparr = np.array(_get_quadrant_map(model, quad, arr_borders=True))
        arr = np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]
        couples_dict[quad] = dict(arr)
    return couples_dict


class Detector:
    def __init__(self, model):
        self.label = model
        self.map = get_map(model)
        self.couples = get_couples(model)
        self.quadrant_keys = [
            *filter(lambda q: set(self.map[q]) != {UNBOND}, [*self.map.keys()])
        ]

    def scintids(self):
        """
        Returns a dict with list values and string keys.
        Keys represents quadrant, values are scintillators ids.
        """
        out = {}
        for quad in self.couples.keys():
            out[quad] = sorted(list(self.couples[quad].keys()))
        return out

    def scintid(self, quad, ch):
        """
        given a channel's quadrant and index,
        returns the scintillator id.

        Args:
            quad: string, quadrant id
            ch: int, channel id

        Returns: int, scintillator id

        """
        if ch in self.couples[quad].keys():
            return ch
        return self.companion(quad, ch)

    def companion(self, quad, ch):
        """
        given a channel's quadrant and index,
        returns the id of the companion cell

        Args:
            quad: string, quadrant id
            ch: int, channel id

        Returns: int, companion channel id

        """
        if ch in self.couples[quad].keys():
            return self.couples[quad][ch]
        companions = {k: v for v, k in self.couples[quad].items()}
        return companions[ch]


# will run some test on detector maps
if __name__ == "__main__":
    TOT_QUAD = 4
    TOT_CH = 32
    TOT_BONDED = 30
    ROWS = 6
    COLS = 5

    for model, map_ in _maps.items():
        # checks for 4 quadrants
        assert len(map_.keys()) == TOT_QUAD
        for quadrant in map_.keys():
            print("testing quadrant {} of {}..".format(quadrant, model))
            channels = [sdd for sdd in map_[quadrant]]
            bonded = [sdd for sdd in map_[quadrant] if sdd != UNBOND]
            # checks for 32 channels
            assert len(channels) == TOT_CH
            # checks for no duplicates in each quadrant
            if len(bonded) != len(set(bonded)):
                print(
                    "duplicated channels quadrant {} of {}".format(
                        quadrant, model
                    )
                )
                print(set(ch for ch in bonded if bonded.count(ch) > 1))
            assert len(bonded) == len(set(bonded))
            # check no entry out of grid
            for row, col in bonded:
                assert row < ROWS
                assert col < COLS
            if not ((model == "dm") and quadrant in ["B", "C", "D"]):
                # checks for 2 unbounded channels
                assert len(set(bonded)) == TOT_BONDED
                # check for all places to be assigned
                for row in range(ROWS):
                    for column in range(COLS):
                        assert (row, column) in bonded
    print("Good news! All tests passed.")
