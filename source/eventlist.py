import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import source.errors as err
from source.constants import PHOTOEL_PER_KEV

s2i = lambda quad: "ABCD".find(str.upper(quad))
i2s = lambda n: chr(65 + n)


def _as_ucid_dataframe(dict_of_df):
    out = pd.concat(
        [
            pd.DataFrame(
                df.values,
                index=df.index.map(lambda x: x + 100 * s2i(key)),
                columns=df.columns,
            )
            for key, df in dict_of_df.items()
        ]
    )
    return out


def _convert_x_events(data):
    out = data[:]
    energies = out["ELECTRONS"] * PHOTOEL_PER_KEV
    out.insert(0, "ENERGY", energies)
    out.drop(columns=["ELECTRONS"])
    return out


def _convert_gamma_events(data, scint_calibrations, couples):
    out = data[:]
    qm = out["QUADID"].map({key: 100 * s2i(key) for key in "ABCD"})
    inverted_couples = {key: {v: k for k, v in d.items()} for key, d in couples.items()}
    companion_to_channel = dict(
        np.concatenate(
            [
                np.array([*inverted_couples[key].items()]) + 100 * s2i(key)
                for key in couples.keys()
            ]
        )
    )
    channel = out["CHN"] + qm
    scint_ucid = channel.map(companion_to_channel).fillna(channel).astype(int)
    ucid_calibs = _as_ucid_dataframe(scint_calibrations)
    if scint_ucid.isin(ucid_calibs.index).all():
        electrons = out["ELECTRONS"]
        light_outs = ucid_calibs.loc[scint_ucid]["light_out"].values
    elif scint_ucid.isin(ucid_calibs.index).any(bool_only=True):
        mask = scint_ucid.isin(ucid_calibs.index)
        electrons = out[mask]["ELECTRONS"]
        light_outs = ucid_calibs.loc[scint_ucid[mask]]["light_out"].values
        uncalibrated_events = out[~mask]
        bad_channels = uncalibrated_events[["CHN", "QUADID"]].drop_duplicates().values.tolist()
        logging.warning(
            "{} ({:.2f} %) events from channels {} were not calibrated.".
            format(
                len(uncalibrated_events),
                100 * len(uncalibrated_events) / len(electrons),
                str([quad + "{:02d}".format(ch) for (ch, quad) in bad_channels])[1:-1],
            )
        )
    else:
        raise err.CalibratedEventlistError("failed event calibration.")
    energies = electrons / light_outs

    out.insert(0, "ENERGY", energies)
    out.drop(columns=["ELECTRONS"])
    return out


def electrons_to_energy(data, scint_calibrations, couples):
    x_events = _convert_x_events(data)
    gamma_events = _convert_gamma_events(data, scint_calibrations, couples)
    out = pd.concat((x_events, gamma_events)).sort_values("TIME").reset_index(drop=True)
    return out


def make_electron_list(
    data, calibrated_sdds, sfit_results, scintillator_couples, nthreads=1,
):
    columns = ["TIME", "ELECTRONS", "EVTYPE", "CHN", "QUADID"]
    types = ["float64", "float32", "U1", "int8", "U1"]
    dtypes = {col: tp for col, tp in zip(columns, types)}
    container = np.recarray(shape=0, dtype=[*dtypes.items()])

    disorganized_events = _get_calibrated_events(
        data, calibrated_sdds, sfit_results, scintillator_couples, nthreads=nthreads,
    )

    for quadrant in disorganized_events.keys():
        x_events, gamma_events = disorganized_events[quadrant]

        xtimes, xenergies, xchannels = x_events.T
        xquadrants = np.array([quadrant] * len(x_events))
        xevtypes = np.array(["X"] * len(x_events))

        stimes, senergies, schannels = gamma_events.T
        squadrants = np.array([quadrant] * len(gamma_events))
        sevtypes = np.array(["S"] * len(gamma_events))

        x_array = np.rec.fromarrays(
            [xtimes, xenergies, xevtypes, xchannels, xquadrants],
            dtype=[*dtypes.items()],
        )
        s_array = np.rec.fromarrays(
            [stimes, senergies, sevtypes, schannels, squadrants],
            dtype=[*dtypes.items()],
        )
        container = np.hstack((container, x_array, s_array))
    out = pd.DataFrame(container).sort_values("TIME").reset_index(drop=True)
    return out


def _get_calibrated_events(
    data, calibrated_sdds, sfit_results, scintillator_couples, nthreads=1
):
    def helper(quadrant):
        couples = scintillator_couples[quadrant]
        calibrated_channels = sfit_results[quadrant].index
        coupled_channels = _get_coupled_channels(calibrated_channels, couples)

        quadrant_data = data[
            (data["QUADID"] == quadrant) & (data["CHN"].isin(coupled_channels))
        ]
        quadrant_data = _insert_electron_column(
            quadrant_data,
            calibrated_sdds[quadrant],
        )

        x_events = _extract_x_events(quadrant_data)
        gamma_events = _extract_gamma_events(
            quadrant_data,
            scintillator_couples[quadrant],
        )

        return quadrant, (x_events, gamma_events)

    results = Parallel(n_jobs=nthreads)(
        delayed(helper)(quad) for quad in sfit_results.keys()
    )
    return {quadrant: value for quadrant, value in results}


def _get_coupled_channels(channels, couples):
    """
    Returns a subset of channels filtered of the "widow" channels

    Args:
        channels: a list of channels
        couples: a list of channels couples (e.g., two elements lists or tuples)

    Returns:
        a list  of channels.
    """
    coupled_channels = []
    for couple in couples.items():
        if all([channel in channels for channel in couple]):
            coupled_channels += [ch for ch in couple]
    return coupled_channels


def _extract_gamma_events(quadrant_data, scintillator_couples):
    assert(np.any(quadrant_data.values))
    gamma_events = quadrant_data[quadrant_data["EVTYPE"] == "S"]
    channels = gamma_events["CHN"]
    companion_to_chn = {k: v for v, k in scintillator_couples.items()}
    same_value_if_coupled = gamma_events["CHN"].map(companion_to_chn).fillna(channels)
    gamma_events = gamma_events.assign(CHN=same_value_if_coupled)

    simultaneous_scintillator_events = gamma_events.groupby(["TIME", "CHN"])
    times, channels = np.array([*simultaneous_scintillator_events.groups.keys()]).T

    electrons_sum = simultaneous_scintillator_events.sum(numeric_only=True)[
        "ELECTRONS"
    ].values
    calibrated_gamma_events = np.column_stack((times, electrons_sum, channels))
    return calibrated_gamma_events


def _extract_x_events(quadrant_data):
    return quadrant_data[quadrant_data["EVTYPE"] == "X"][
        ["TIME", "ELECTRONS", "CHN"]
    ].values


def _insert_electron_column(data, calibrated_sdds):
    adcs = data["ADC"]
    chns = data["CHN"]
    offsets = calibrated_sdds.loc[chns]["offset"].values
    gains = calibrated_sdds.loc[chns]["gain"].values

    electrons = (adcs - offsets) / gains / PHOTOEL_PER_KEV
    data.insert(0, "ELECTRONS", electrons)
    return data


def add_evtype_tag(data, couples):
    """
    inplace add event type (X or S) column
    """
    qm = data["QUADID"].map({key: 100 * s2i(key) for key in "ABCD"})
    chm_dict = dict(
        np.concatenate(
            [
                np.array([*couples[key].items()]) + 100 * s2i(key)
                for key in couples.keys()
            ]
        )
    )
    chm = data["CHN"] + qm
    data.insert(
        loc=len(data.columns),
        column="EVTYPE",
        value=(
            data.assign(CHN=chm.map(chm_dict).fillna(chm))
            .duplicated(["SID", "CHN"], keep=False)
            .map({False: "X", True: "S"})
            .astype("string")
        ),
    )
    return data


def perchannel_counts(data, channels):
    dict_ = {}
    for quad in channels.keys():
        quaddata = data[data["QUADID"] == quad]
        for ch in channels[quad]:
            counts = len(quaddata[(quaddata["CHN"] == ch)])
            dict_.setdefault(quad, {})[ch] = counts

    out = {
        k: pd.DataFrame(dict_[k], index=["counts"],).T.rename_axis("channel")
        for k in dict_
    }
    return out


def filter_spurious(data):
    return data[(data["NMULT"] < 2) | ((data["NMULT"] == 2) & (data["EVTYPE"] == "S"))]


def filter_delay(data, hold_time):
    unique_times = data.TIME.unique()
    bad_events = unique_times[np.where(np.diff(unique_times) < hold_time)[0] + 1]
    return data.drop(data.index[data["TIME"].isin(bad_events)]).reset_index(drop=True)


def infer_onchannels(data):
    out = {}
    for quad in "ABCD":
        onchs = np.unique(data[data["QUADID"] == quad]["CHN"])
        if onchs.any():
            out[quad] = onchs.tolist()
    return out
