from math import floor
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits as fitsio

i2s = (lambda n: chr(65 + n))
s2i = (lambda asic: "ABCD".find(str.upper(asic)))


def pandas_from(fits: Path):
    fits_path = Path(fits)

    with fitsio.open(fits_path) as fits_file:
        # would be nice to just pd.DataFrame(fits_file[1].data) but endians
        df = pd.DataFrame(np.array(fits_file[-1].data).byteswap().newbyteorder())
    start_t = floor(df[df['TIME'] > 1].iloc[0]['TIME']) - 1
    df.loc[df['TIME'] < 1, 'TIME'] += start_t

    columns = ['ADC', 'CHN', 'QUADID', 'NMULT', 'TIME', 'EVTID']
    types = ['int32', 'int8', 'object', 'int8', 'float32', 'int32']
    dtypes = {col: tp for col, tp in zip(columns, types)}

    temp = np.concatenate([df[['ADC' + i, 'CHANNEL' + i, 'QUADID', 'NMULT', 'TIME', 'EVTID']] for i in '012345'])
    temp = temp[temp[:, 0] > 0]
    temp = temp[temp[:, -1].argsort()]
    df = pd.DataFrame(temp, columns=columns)
    df = df.assign(QUADID=df['QUADID'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})).astype(dtypes).astype(dtypes)
    return df


def add_evtype_tag(data, couples):
    """
    inplace add event type (X or S) column
    :param data:
    :return:
    """
    data['CHN'] = data['CHN'] + 1
    qm = data['QUADID'].map({key: 100 ** s2i(key) for key in 'ABCD'})
    chm_dict = dict(np.concatenate([(couples[key] + 1) * 100**s2i(key) for key in 'ABCD']))
    chm = data['CHN']*qm
    data.insert(loc=3, column='EVTYPE', value=(data
                                               .assign(CHN=chm.map(chm_dict).fillna(chm))
                                               .duplicated(['TIME', 'CHN'], keep=False)
                                               .map({False: 'X', True: 'S'})
                                               .astype('string')))
    data['CHN'] = data['CHN'] - 1
    return data