import pandas as pd
import numpy as np
from source.inventory import get_quadrant_map

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


def filter_spurious(data):
    return data[(data["NMULT"] < 2) | ((data["NMULT"] == 2) & (data["EVTYPE"] == "S"))]


def filter_delay(data, hold_time):
    unique_times = data.TIME.unique()
    bad_events = unique_times[np.where(np.diff(unique_times) < hold_time)[0] + 1]
    return data.drop(data.index[data['TIME'].isin(bad_events)]).reset_index(drop=True)


def infer_onchannels(data):
    out = {}
    for quad in 'ABCD':
        onchs = np.unique(data[data['QUADID'] == quad]['CHN'])
        if onchs.any():
            out[quad] = onchs
    return out


def get_quad_couples(quad):
    qmaparr = np.array(get_quadrant_map(quad))
    return np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]


def get_couples():
    return {q: get_quad_couples(q) for q in "ABCD"}