import pandas as pd
import numpy as np
from assets import detectors
from source.errors import UnknownModelError

s2i = (lambda quad: "ABCD".find(str.upper(quad)))
i2s = (lambda n: chr(65 + n))


def add_evtype_tag(data, couples):
    """
    inplace add event type (X or S) column
    """
    data['CHN'] = data['CHN'] + 1
    qm = data['QUADID'].map({key: 100 ** s2i(key) for key in 'ABCD'})
    chm_dict = dict(np.concatenate([(couples[key] + 1) * 100 ** s2i(key) for key in couples.keys()]))
    chm = data['CHN'] * qm
    data.insert(loc=3, column='EVTYPE', value=(data
                                               .assign(CHN=chm.map(chm_dict).fillna(chm))
                                               .duplicated(['SID', 'CHN'], keep=False)
                                               .map({False: 'X', True: 'S'})
                                               .astype('string')))
    data['CHN'] = data['CHN'] - 1
    return data


def filter_delay(data, hold_time):
    unique_times = data.TIME.unique()
    bad_events = unique_times[np.where(np.diff(unique_times) < hold_time)[0] + 1]
    return data.drop(data.index[data['TIME'].isin(bad_events)]).reset_index()


def infer_onchannels(data: pd.DataFrame):
    out = {}
    for quad in 'ABCD':
        onchs = np.unique(data[data['QUADID'] == quad]['CHN'])
        if onchs.any():
            out[quad] = onchs
    return out


def get_qmap(model: str, quad: str, arr_borders: bool = True):
    if model == 'fm1':
        detector_map = detectors.fm1
    else:
        raise UnknownModelError("unknown model.")

    if quad in ['A', 'B', 'C', 'D']:
        arr = detector_map[quad]
    else:
        raise ValueError("Unknown quadrant key. Allowed keys are A,B,C,D")

    if arr_borders:
        return tuple(map(lambda x: (x[0] + int(x[0] / 2), x[1]), arr))
    return arr


def get_channels(model: str, quad: str):
    return [ch for ch, _ in enumerate(get_qmap(model, quad)) if ch is not detectors.UNBOND]


def get_quad_couples(model: str, quad: str):
    qmaparr = np.array(get_qmap(model, quad))
    return np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]


def get_couples(model: str):
    return {q: get_quad_couples(model, q) for q in "ABCD"}