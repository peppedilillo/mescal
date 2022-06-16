from math import floor
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits as fitsio
from assets import detectors


class SourceNotFoundError(Exception):
    """An error while parsing calib sources."""


class UnknownModelError(Exception):
    """An error when querying an unsupported detector."""


def infer_onchannels(data: pd.DataFrame):
    out = {}
    for asic in 'ABCD':
        onchs = np.unique(data[data['QUADID'] == asic]['CHN'])
        if onchs.any():
            out[asic] = onchs
    return out


def write_report_to_excel(result_df, path):
    with pd.ExcelWriter(path) as output:
        for asic in result_df.keys():
            result_df[asic].to_excel(output, sheet_name=asic, engine='xlsxwriter', encoding='utf8')
    return True


def read_report_from_excel(from_path):
    return pd.read_excel(from_path, index_col=0, sheet_name=None)


def write_report_to_csv(result_df, path):
    for quad, df in result_df.items():
        df.to_csv(path(quad, asic), sep=';')
    return True


def read_report_from_csv(from_path):
    pass


def write_report_to_fits(result_df, path):
    header = fits.PrimaryHDU()
    output = fits.HDUList([header])
    for asic in result_df.keys():
        table_asic = fits.BinTableHDU.from_columns(result_df[asic].to_records(), name=asic)
        output.append(table_asic)
    output.writeto(path, overwrite=True)
    return True


def read_report_from_fits(path):
    pass


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
