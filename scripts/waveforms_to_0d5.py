from math import ceil
from pathlib import Path
import argparse

import astropy.io.fits as fitsio
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("quadrant", action="store", type=int)
parser.add_argument("outfile", action="store", type=str)
parser.add_argument("infiles", action="store", type=str, nargs="+")

# refers to CAEN DT5740
NCH = 32
DEFAULT_PARS = {
    "time_unit_s": 8 * 10 ** -9,
    "sample_dur_us": 0.016,
    "mux_time_us": 10.0,
    "offset_from_start_us": 0.64,
    "offset_from_stop_us": 3.84,
}
DEFAULT_CHANNELS = {
    "waveforms": "CH00",
    "start_conversion": "CH08",
    "trigger_mux_out": "CH12",
}
DEFAULT_MASKS = tuple([(-np.inf, -np.inf)] * NCH)
DEAFULT_PARALLEL_NJOBS = -1  # all workers


def get_meta(filepath):
    with fitsio.open(filepath) as hdul:
        decimation = hdul[1].header["DECIMFAC"]
        trigger_sample = hdul[1].header["PRETRG"]
    return decimation, trigger_sample


def get_times(filepath):
    with fitsio.open(filepath) as hdul:
        data = hdul[1].data
        times = data["TIME"]
        # first event is bad
        times = times[1:]
    return times


def get_waveforms(filepath, wf_ch, tmux_ch):
    with fitsio.open(filepath) as hdul:
        data = hdul[1].data
        waveforms = data.field(wf_ch)
        trigmuxs = data.field(tmux_ch)
        # first event is bad
        waveforms = waveforms[1:]
        trigmuxs = trigmuxs[1:]
    return waveforms, trigmuxs


def table_from_wfs(
    filepath: Path,
    wf_ch=DEFAULT_CHANNELS["waveforms"],
    tmux_ch=DEFAULT_CHANNELS["trigger_mux_out"],
    time_unit_s=DEFAULT_PARS["time_unit_s"],
    sample_dur_us=DEFAULT_PARS["sample_dur_us"],
    mux_time_us=DEFAULT_PARS["mux_time_us"],
    offset_from_start_us=DEFAULT_PARS["offset_from_start_us"],
    offset_from_stop_us=DEFAULT_PARS["offset_from_stop_us"],
    nevents="all",
    n_jobs=DEAFULT_PARALLEL_NJOBS,
):
    print("Analyzing file {}..".format(filepath.name))
    decimation, trigger_sample = get_meta(filepath)
    waveforms, trigmux = get_waveforms(filepath, wf_ch, tmux_ch)
    nevents = (lambda n: len(waveforms) if n == "all" else n)(nevents)

    times = get_times(filepath) * time_unit_s
    events = table_make(
        waveforms,
        trigmux,
        decimation,
        trigger_sample,
        sample_dur_us,
        mux_time_us,
        offset_from_start_us,
        offset_from_stop_us,
        nevents,
        n_jobs,
    )
    return times, events


def get_mux_intervals(
    decimation,
    trigger_sample,
    sample_dur_us,
    mux_time_us,
    offset_from_start_us,
    offset_from_stop_us,
):
    sampling_time_us = sample_dur_us * 2 ** decimation
    mux_intervals = [
        (
            ceil(
                trigger_sample
                + offset_from_start_us / sampling_time_us
                + i * mux_time_us / sampling_time_us
            ),
            ceil(
                trigger_sample
                - offset_from_stop_us / sampling_time_us
                + (i + 1) * mux_time_us / sampling_time_us
            ),
        )
        for i in range(NCH)
    ]
    return mux_intervals


def table_make(
    waveforms,
    trgmuxs,
    decimation,
    trigger_sample,
    sample_dur_us,
    mux_time_us,
    offset_from_start_us,
    offset_from_stop_us,
    nevents,
    n_jobs,
):
    def parhelper(i, waveform, trgmux, mux_intervals):
        trg_mux_max = np.max(trgmux)
        out = [
            np.mean(waveform[idx_start:idx_stop])
            if 2 * np.mean(trgmux[idx_start:idx_stop]) > trg_mux_max
            else np.nan
            for idx_start, idx_stop in mux_intervals
        ]
        return [i] + out

    mux_intervals = get_mux_intervals(
        decimation,
        trigger_sample,
        sample_dur_us,
        mux_time_us,
        offset_from_start_us,
        offset_from_stop_us,
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(parhelper)(i, waveforms[i], trgmuxs[i], mux_intervals)
        for i in range(nevents)
    )
    event_list = np.vstack(results)
    event_list = event_list[np.argsort(event_list[:, 0])]
    return event_list[:, 1:]


def transform_table(big_table):
    # e.g.  big_table = [[np.nan, 2.1, np.nan, 1.1],
    #                    [3.3, np.nan, np.nan, 6.6]])
    # sort the big table by (ADC) values
    # np.nans go to the right of the matrix
    # big_table_sortedADCs = [[1.1, 2.1, nan],
    #                         [3.3, 6.6, nan]]
    big_table_sortedADCs = np.sort(big_table, axis=1)[:, :6]  # [:, :3]
    # argsort the big table by (ADC) values
    # untriggered channels (np.nan) go to the right of the matrix
    # big_table_idCHNs_sortedADCs = [[3, 1, 0],
    #                                [0, 3, 1]]
    big_table_idCHNs_sortedADCs = np.argsort(big_table, axis=1)[
        :, :6
    ]  # [:, :3]
    # replace channels IDs of untriggered channels with 32
    # big_table_idCHNs_sortedADCs = [[3, 1, 32],
    #                                [0, 3, 32]]
    big_table_idCHNs_sortedADCs[np.isnan(big_table_sortedADCs)] = +32
    # replace ADCs value of untriggered channels with -1.0
    # big_table_sortedADCs = [[1.1, 2.1, -1.0],
    #                         [3.3, 6.6, -1.0]]
    big_table_sortedADCs[np.isnan(big_table_sortedADCs)] = -1
    # sort again the tables, this time by the channel IDs.
    # all untriggered channels stays in their place, since their value is 32.
    # big_table_idCHNs_sortedADCs = [[1, 3, -1],
    #                                [0, 3, -1]]
    # big_table_sortedADCs = [[2.1, 1.1, -1.0],
    #                         [3.3, 6.6, -1.0]]
    sorting_by_idCHNs = np.argsort(big_table_idCHNs_sortedADCs, axis=1)
    big_table_idCHNs_sortedADCs[big_table_idCHNs_sortedADCs == +32] = -1
    big_table_idCHNs_sortedADCs = np.take_along_axis(
        big_table_idCHNs_sortedADCs, sorting_by_idCHNs, axis=1,
    )
    big_table_sortedADCs = np.take_along_axis(
\
def table0d5_from_wftable(quadid, times, big_table):
    # evtids = np.argwhere(~np.isnan(big_table))[:,0]
    nmults = np.sum(~np.isnan(big_table), axis=1)
    quadids = np.zeros((len(big_table), 1)) + quadid
    small_table = transform_table(big_table)

    table0d5 = np.column_stack((times, nmults, quadids, small_table))
    return table0d5


def dataframe_from_table0d5(table):
    df = pd.DataFrame(
        table,
        columns=[
            "TIME",
            "NMULT",
            "QUADID",
            "CHANNEL0",
            "ADC0",
            "CHANNEL1",
            "ADC1",
            "CHANNEL2",
            "ADC2",
            "CHANNEL3",
            "ADC3",
            "CHANNEL4",
            "ADC4",
            "CHANNEL5",
            "ADC5",
        ],
    ).astype(
        {
            "TIME": "float64",
            "NMULT": "int16",
            "QUADID": "int16",
            "CHANNEL0": "int16",
            "ADC0": "int32",
            "CHANNEL1": "int16",
            "ADC1": "int32",
            "CHANNEL2": "int16",
            "ADC2": "int32",
            "CHANNEL3": "int16",
            "ADC3": "int32",
            "CHANNEL4": "int16",
            "ADC4": "int32",
            "CHANNEL5": "int16",
            "ADC5": "int32",
        }
    )
    return df


def save_to_fits(dataframe, filepath):
    header_fits = fitsio.PrimaryHDU()
    output_fits = fitsio.HDUList([header_fits])
    table_fits = fitsio.BinTableHDU.from_columns(
        dataframe.to_records(index=False,),
        name="CAEN DT5740 mock-up HERMES L0d5.",
    )
    output_fits.append(table_fits)
    output_fits.writeto(filepath, overwrite=True)
    return filepath


if __name__ == "__main__":
    args = parser.parse_args()
    tables = []
    last_time = 0
    for filepath in args.infiles:
        times, wftable = table_from_wfs(Path(filepath))
        tables.append(table0d5_from_wftable(args.quadrant, times + last_time, wftable))
        last_time += times[-1]
    dataframe = dataframe_from_table0d5(np.vstack(tables))
    save_to_fits(dataframe, args.outfile)
    print("Done!")

    #table = np.vstack([
    #    table0d5_from_wftable(
    #        args.quadrant, *table_from_wfs(
    #            Path(filepath)))
    #    for filepath in args.infiles])
    #dataframe = dataframe_from_table0d5(table)
    #save_to_fits(dataframe, args.outfile)

    #QUADID = 1
    #filepath = Path(
    #    "D:/IAPSDATA/FM2/20220927/20220927_03_FM2_T19deg_QUADA_THR110_DAC120_SingleCH_HV123_no_CH02_03_04_06_09_24_25_Cd109.fits"
    #)
    #out_filepath = Path(
    #    "D:/IAPSDATA/FM2/20220927/0d5_20220927_03_FM2_T19deg_QUADA_THR110_DAC120_SingleCH_HV123_no_CH02_03_04_06_09_24_25_Cd109.fits"
    #)
#
    #tic = time.time()
    #times = get_times(filepath) * DEFAULT_PARS["time_unit_s"]
    #times, big_table = table_from_wfs(filepath)
    #small_table = table0d5_from_wftable(times, big_table)
    #small_dataframe = dataframe_from_table0d5(small_table)
    #save_to_fits(small_dataframe, out_filepath)
    #toc = time.time()
    #print("execution time: {:.2f}s".format(toc - tic))
