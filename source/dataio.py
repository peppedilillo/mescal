from math import floor
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits as fitsio
from assets import detectors


class UnknownModelError(Exception):
    """An error when querying an unsupported detector."""


def infer_onchannels(data: pd.DataFrame):
    out = {}
    for quad in 'ABCD':
        onchs = np.unique(data[data['QUADID'] == quad]['CHN'])
        if onchs.any():
            out[quad] = onchs
    return out


def write_report_to_excel(result_df, path):
    with pd.ExcelWriter(path) as output:
        for quad in result_df.keys():
            result_df[quad].to_excel(output, sheet_name=quad, engine='xlsxwriter', encoding='utf8')
    return True


def read_report_from_excel(from_path):
    return pd.read_excel(from_path, index_col=0, sheet_name=None)


def write_report_to_csv(result_df, path):
    for quad, df in result_df.items():
        df.to_csv(path(quad, quad))
    return True


def read_report_from_csv(from_path):
    pass


def write_report_to_fits(result_df, path):
    header = fitsio.PrimaryHDU()
    output = fitsio.HDUList([header])
    for quad in result_df.keys():
        table_quad = fitsio.BinTableHDU.from_columns(result_df[quad].to_records(), name=quad)
        output.append(table_quad)
    output.writeto(path, overwrite=True)
    return True


def read_report_from_fits(path):
    pass


def write_eventlist_to_fits(eventlist, path):
    header = fitsio.PrimaryHDU()
    output = fitsio.HDUList([header])
    table_quad = fitsio.BinTableHDU.from_columns(
        eventlist.to_records(index=False, column_dtypes={'EVTYPE':'U1', 'CHN': 'i8', 'QUADID':'U1'}),
        name='Event list')
    output.append(table_quad)
    output.writeto(path, overwrite=True)
    return True


def get_qmap(model:str, quad: str, arr_borders: bool = True):
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


def get_channels(model:str, quad: str):
    return [ch for ch, _ in enumerate(get_qmap(model, quad)) if ch is not UNBOND]


def get_couples(model:str, quad: str):
    qmaparr = np.array(get_qmap(model, quad))
    return np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]


def pandas_from(fits: Path):
    fits_path = Path(fits)

    with fitsio.open(fits_path) as fits_file:
        df = pd.DataFrame(np.array(fits_file[-1].data).byteswap().newbyteorder())
    start_t = floor(df[df['TIME'] > 1].iloc[0]['TIME']) - 1
    df.loc[df['TIME'] < 1, 'TIME'] += start_t
    df = df.reset_index(level=0).rename({'index': 'SID'}, axis='columns')

    columns = ['ADC', 'CHN', 'QUADID', 'NMULT', 'TIME', 'SID']
    types = ['int32', 'int8', 'string', 'int8', 'float64', 'int32']
    dtypes = {col: tp for col, tp in zip(columns, types)}

    temp = np.concatenate([df[[ 'ADC' + i, 'CHANNEL' + i, 'QUADID', 'NMULT', 'TIME', 'SID']] for i in '012345'])
    temp = temp[temp[:, 0] > 0]
    temp = temp[temp[:, -1].argsort()]
    df = pd.DataFrame(temp, columns=columns)
    df = df.assign(QUADID=df['QUADID'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})).astype(dtypes)
    return df
