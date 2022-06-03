from math import floor
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits as fitsio
from source import upaths

i2s = (lambda n: chr(65 + n))
s2i = (lambda asic: "ABCD".find(str.upper(asic)))


def get_from(fitspath: Path) -> pd.DataFrame:
    cached = upaths.CACHEDIR.joinpath(fitspath.name).with_suffix('.pkl.gz')
    if cached.is_file():
        print("Reading from cache..")
        df = pd.read_pickle(cached)
    elif fitspath.is_file():
        print("Reading from fits..")
        df = pandas_from(fitspath)
        print("Saving to cache..")
        df.to_pickle(cached)
    else:
        raise FileNotFoundError('Could not locate input datafile.')
    return df


def infer_onchannels(data: pd.DataFrame, quad: str):
    return np.unique(data[data['QUADID'] == quad]['CHN'])


def pandas_from(fits: Path) -> pd.DataFrame:
    fits_path = Path(fits)

    with fitsio.open(fits_path) as fits_file:
        # would be nice to just pd.DataFrame(fits_file[1].data) but endians
        df = pd.DataFrame(np.array(fits_file[1].data).byteswap().newbyteorder())
    start_t = floor(df[df['TIME'] > 1].iloc[0]['TIME']) - 1
    df.loc[df['TIME'] < 1, 'TIME'] += start_t

    columns = ['ADC', 'CHN', 'QUADID', 'NMULT', 'TIME', 'EVTID']
    types = ['int32', 'int8', 'object', 'int8', 'float32', 'int32']
    # types = ['int32', 'int8', 'int8', 'int8', 'float32', 'int32']
    dtypes = {col: tp for col, tp in zip(columns, types)}

    temp = np.concatenate([df[['ADC' + i, 'CHANNEL' + i, 'QUADID', 'NMULT', 'TIME', 'EVTID']] for i in '012345'])
    temp = temp[temp[:, 0] > 0]
    temp = temp[temp[:, -1].argsort()]
    df = pd.DataFrame(temp, columns=columns)
    df = df.assign(QUADID=df['QUADID'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})).astype(dtypes).astype(dtypes)
    return df


def add_evtype_flag_to(quad_data: pd.DataFrame, couples) -> pd.DataFrame:
    d = dict(couples)
    quad_data.insert(loc=3, column='EVTYPE', value=(quad_data
                                                    .assign(CHN=quad_data['CHN'].map(d).fillna(quad_data['CHN']))
                                                    .duplicated(['EVTID', 'CHN'], keep=False)
                                                    .map({False: 'X', True: 'S'})
                                                    .astype('string')))
    return quad_data
