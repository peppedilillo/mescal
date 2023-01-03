from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits as fitsio

from source.errors import FormatNotSupportedError


def get_writer(fmt):
    if fmt == "xslx":
        return write_report_to_excel
    elif fmt == "fits":
        return write_report_to_fits
    elif fmt == "csv":
        return write_report_to_csv
    else:
        raise FormatNotSupportedError("write format not supported")


def write_report_to_excel(result_df, path):
    with pd.ExcelWriter(path) as output:
        for quad in result_df.keys():
            result_df[quad].to_excel(
                output,
                sheet_name=quad,
                engine="xlsxwriter",
            )
    return True


def read_report_from_excel(from_path, kind):
    if kind == "calib":
        return pd.read_excel(from_path, index_col=0, sheet_name=None)
    elif kind == "peaks":
        return pd.read_excel(from_path, header=[0, 1], index_col=0, sheet_name=None)
    elif kind == "fits":
        return pd.read_excel(from_path, header=[0, 1], index_col=0, sheet_name=None)
    else:
        raise ValueError("kind must be either 'calib', 'peaks', or 'fits'.")


def write_report_to_fits(result_df, path):
    header = fitsio.PrimaryHDU()
    output = fitsio.HDUList([header])
    for quad in result_df.keys():
        table_quad = fitsio.BinTableHDU.from_columns(
            result_df[quad].to_records(), name="Quadrant " + quad
        )
        output.append(table_quad)
    output.writeto(path.with_suffix(".fits"), overwrite=True)
    return True


def read_report_from_fits(path):
    pass


def write_report_to_csv(result_df, path):
    for quad, df in result_df.items():
        df.to_csv(
            path.with_name(path.stem + "_quad{}".format(quad)).with_suffix(".csv")
        )
    return True


def read_report_from_csv(from_path):
    pass


def write_eventlist_to_fits(eventlist, path):
    header = fitsio.PrimaryHDU()
    output = fitsio.HDUList([header])
    table_quad = fitsio.BinTableHDU.from_columns(
        eventlist.to_records(
            index=False,
            column_dtypes={"EVTYPE": "U1", "CHN": "i8", "QUADID": "U1"},
        ),
        name="Event list",
    )
    output.append(table_quad)
    output.writeto(path, overwrite=True)
    return True


def pandas_from_LV0d5(fits: Path):
    fits_path = Path(fits)
    with fitsio.open(fits_path) as fits_file:
        fits_data = fits_file[-1].data
        df = pd.DataFrame(np.array(fits_data).byteswap().newbyteorder())
    # fixes first buffer missing ABT
    start_t = floor(df[df["TIME"] > 1].iloc[0]["TIME"]) - 1
    df.loc[df["TIME"] < 1, "TIME"] += start_t

    df = df.reset_index(level=0).rename({"index": "SID"}, axis="columns")
    columns = ["ADC", "CHN", "QUADID", "NMULT", "TIME", "SID"]
    types = ["int32", "int8", "string", "int8", "float64", "int32"]
    dtypes = {col: tp for col, tp in zip(columns, types)}

    temp = np.concatenate(
        [
            df[["ADC" + i, "CHANNEL" + i, "QUADID", "NMULT", "TIME", "SID"]]
            for i in "012345"
        ]
    )
    temp = temp[temp[:, 0] > 0]
    temp = temp[temp[:, -1].argsort()]
    df = pd.DataFrame(temp, columns=columns)
    df = df.assign(QUADID=df["QUADID"].map({0: "A", 1: "B", 2: "C", 3: "D"})).astype(
        dtypes
    )
    return df
