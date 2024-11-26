import logging

from joblib import delayed
from joblib import Parallel
import numpy as np
import pandas as pd

from source.constants import PHOTOEL_PER_KEV
from source.detectors import Detector
from source.detectors import get_couples
import source.errors as err

s2i = lambda quad: "ABCD".find(str.upper(quad))
i2s = lambda n: chr(65 + n)


def _as_ucid_dataframe(dict_of_df) -> pd.DataFrame:
    """
    Transforms scintillator calibrations from a dictionary of dataframe to a
    single dataframe, with uniquely defined indeces (e.g. C21 --> 221, A11 -->11
    and so on). Returns an empty dataframe if an empty dictionary is passed.
    """
    out = pd.concat(
        # prevents an error and ensues returning empty df if dict_of_df is empty
        [pd.DataFrame()]
        + [
            pd.DataFrame(
                df.values,
                index=df.index.map(lambda x: x + 100 * s2i(key)),
                columns=df.columns,
            )
            for key, df in dict_of_df.items()
        ]
    )
    return out


def preprocess(
    data,
    model,
    filter_retrigger=20 * 10**-6,
    filter_spurious=True,
    console=None,
):
    """
    This is in-place, meaning that will add a column to data.
    """
    couples = get_couples(model)
    data = add_evtype_tag(data, couples)
    waste = pd.DataFrame()
    events_pre_filter = len(data)
    if console:
        console.log(":white_check_mark: Tagged X and S events.")
    if filter_retrigger:
        data, waste = delay_filter(data, filter_retrigger)
    if filter_spurious:
        data, waste = spurious_filter(data)
        waste = pd.concat((waste, waste))
    filtered = 100 * (events_pre_filter - len(data)) / events_pre_filter
    if filtered and console:
        console.log(
            ":white_check_mark: Filtered {:.1f}% of the events.".format(filtered)
        )
    return data, waste


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


def perchannel_counts(data, channels, key="all"):
    dict_ = {}
    for quad in channels.keys():
        if key == "all":
            quaddata = data[data["QUADID"] == quad]
        elif key == "x":
            quaddata = data[(data["QUADID"] == quad) & (data["EVTYPE"] == "X")]
        elif key == "s":
            quaddata = data[(data["QUADID"] == quad) & (data["EVTYPE"] == "S")]
        else:
            raise ValueError("wrong event type key.")

        for ch in channels[quad]:
            counts = len(quaddata[(quaddata["CHN"] == ch)])
            dict_.setdefault(quad, {})[ch] = counts

    out = {
        k: pd.DataFrame(
            dict_[k],
            index=["counts"],
        ).T.rename_axis("channel")
        for k in dict_
    }
    return out


def find_widows(on_channels: dict, model: Detector):
    widows = {}
    for quad in on_channels:
        widows[quad] = []
        for ch in on_channels[quad]:
            try:
                companion = model.companion(quad, ch)
            except KeyError:
                logging.warning(f"Cant find companion for channel {quad}{ch:02d}.")
                continue
            if companion not in on_channels[quad]:
                widows[quad].append(ch)
    return widows


def spurious_filter(data):
    mask = (data["NMULT"] == 1) | ((data["NMULT"] == 2) & (data["EVTYPE"] == "S"))
    cleaned_data = data[mask]
    waste = data[~mask]
    return cleaned_data, waste


def delay_filter(data, hold_time):
    unique_times = data.TIME.unique()
    bad_events = unique_times[np.where(np.diff(unique_times) < hold_time)[0] + 1]
    mask = ~(data["TIME"].isin(bad_events))
    cleaned_data = data[mask]
    waste = data[~mask]
    return cleaned_data, waste


def filter_channels(data, channels):
    widows_list = [(quad, ch) for quad in channels.keys() for ch in channels[quad]]
    mask = data.apply(lambda row: (row["QUADID"], row["CHN"]) in widows_list, axis=1)
    cleaned_data = data[~mask]
    waste = data[mask]
    return cleaned_data, waste


def infer_onchannels(data):
    out = {}
    for quad in "ABCD":
        onchs = np.unique(data[data["QUADID"] == quad]["CHN"])
        if onchs.any():
            out[quad] = onchs.tolist()
    return out


def timehist(data):
    """
    Makes histograms of counts in time of a channel with safeguards against
    bad time data.
    This function is curried beacuse partial application are useful
    to make histograms for channels in different quadrants in parallel.
    See timehist_quadch for an example interface.
    """

    def timehist_filter_outliers(outliers):
        """Remove time outliers"""

        def timehist_filter_quadrant(quad):
            """Throws away events not on quad"""

            def timehist_filter_channel(ch):
                """Throw away events not on channel"""

                def timehist_histogram(binning):
                    """Makes an histogram"""
                    num_intervals = int((max_ - min_) / binning + 1)
                    counts, bins = np.histogram(
                        data[mask_quadrant & mask_channel]["TIME"].values,
                        range=(min_, min_ + num_intervals * binning),
                        bins=num_intervals,
                    )
                    return counts, bins

                mask_channel = data["CHN"] == ch
                return timehist_histogram

            mask_quadrant = data["QUADID"] == quad
            return timehist_filter_channel

        if outliers:
            min_, max_ = np.quantile(data["TIME"], [0.01, 0.99])
        else:
            min_, max_ = data["TIME"].min(), data["TIME"].max()
        return timehist_filter_quadrant

    return timehist_filter_outliers


def timehist_quadch(data, quad, ch, binning, neglect_outliers):
    """Returns an histograms of counts observed by (quad, ch) removing entries
    with non-sense time information."""
    return timehist(data)(neglect_outliers)(quad)(ch)(binning)


def timehist_all(data, binning, neglect_outliers):
    if len(data) <= 0:
        raise err.BadDataError("Empty data.")

    if neglect_outliers:
        min_, max_ = np.quantile(data["TIME"], [0.01, 0.99])
    else:
        min_, max_ = data["TIME"].min(), data["TIME"].max()
    num_intervals = int((max_ - min_) / binning + 1)
    counts, bins = np.histogram(
        data["TIME"].values,
        range=(min_, min_ + num_intervals * binning),
        bins=num_intervals,
    )
    return counts, bins


def _convert_x_events(data):
    out = data[data["EVTYPE"] == "X"]
    energies = out["ELECTRONS"] * PHOTOEL_PER_KEV
    out.insert(0, "ENERGY", energies)
    out.drop(columns=["ELECTRONS"])
    return out


def _convert_gamma_events(data, scint_calibrations, couples):
    out = data[data["EVTYPE"] == "S"]
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

    if np.all(scint_ucid.isin(ucid_calibs.index).values):
        # all the scintillators were successfully calibrated
        electrons = out["ELECTRONS"]
        light_outs = ucid_calibs.loc[scint_ucid]["light_out"].values

    elif np.any(scint_ucid.isin(ucid_calibs.index).values):
        # some scintillators could not be calibrated for whatever reasons,
        # hence we drop S events from these. Note that this will not
        # filter widow events, since these are not "S" but definition.;
        mask = scint_ucid.isin(ucid_calibs.index)
        electrons = out[mask]["ELECTRONS"]
        light_outs = ucid_calibs.loc[scint_ucid[mask]]["light_out"].values
        uncalibrated_events = out[~mask]
        bad_channels = (
            uncalibrated_events[["CHN", "QUADID"]].drop_duplicates().values.tolist()
        )
        logging.warning(
            "{} ({:.2f} %) events from channels {} were not calibrated.".format(
                len(uncalibrated_events),
                100 * len(uncalibrated_events) / len(electrons),
                str([quad + "{:02d}".format(ch) for (ch, quad) in bad_channels])[1:-1],
            )
        )

    else:
        # we get here if we have no calibrated scintillators.
        # TODO: could be improved to return something empty so that we can at least
        #       return the calibrated X events.
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
    data,
    calibrated_sdds,
    detector,
    nthreads=1,
):
    columns = ["TIME", "ELECTRONS", "EVTYPE", "CHN", "QUADID"]
    types = ["float64", "float32", "U1", "int8", "U1"]
    dtypes = {col: tp for col, tp in zip(columns, types)}
    container = np.recarray(shape=0, dtype=[*dtypes.items()])

    disorganized_events = _get_calibrated_events(
        data,
        calibrated_sdds,
        detector.couples,
        nthreads=nthreads,
    )

    for quadrant in disorganized_events.keys():
        x_events, gamma_events = disorganized_events[quadrant]

        if np.any(x_events):
            xtimes, xenergies, xchannels = x_events.T
            xquadrants = np.array([quadrant] * len(x_events))
            xevtypes = np.array(["X"] * len(x_events))
            x_array = np.rec.fromarrays(
                [xtimes, xenergies, xevtypes, xchannels, xquadrants],
                dtype=[*dtypes.items()],
            )
            container = np.hstack((container, x_array))

        if np.any(gamma_events):
            stimes, senergies, schannels = gamma_events.T
            squadrants = np.array([quadrant] * len(gamma_events))
            sevtypes = np.array(["S"] * len(gamma_events))
            s_array = np.rec.fromarrays(
                [stimes, senergies, sevtypes, schannels, squadrants],
                dtype=[*dtypes.items()],
            )
            container = np.hstack((container, s_array))

    out = pd.DataFrame(container).sort_values("TIME").reset_index(drop=True)
    return out


def _get_calibrated_events(data, calibrated_sdds, scintillator_couples, nthreads=1):
    def helper(quadrant):
        couples = scintillator_couples[quadrant]
        fitted_calibrated_channels = list(set(calibrated_sdds[quadrant].index))
        channels = _get_coupled_channels(fitted_calibrated_channels, couples)

        quadrant_data = data[
            (data["QUADID"] == quadrant) & (data["CHN"].isin(channels))
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
        delayed(helper)(quad) for quad in calibrated_sdds.keys()
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
    if not np.any(quadrant_data.values):
        return np.array([])
    gamma_events = quadrant_data[quadrant_data["EVTYPE"] == "S"]
    channels = gamma_events["CHN"]
    companion_to_chn = {k: v for v, k in scintillator_couples.items()}
    same_value_if_coupled = gamma_events["CHN"].map(companion_to_chn).fillna(channels)
    gamma_events = gamma_events.assign(CHN=same_value_if_coupled)

    if not np.any(gamma_events.values):
        return np.array([])
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
    assert np.all(np.isin(np.unique(chns), calibrated_sdds.index))
    offsets = calibrated_sdds.loc[chns]["offset"].values
    gains = calibrated_sdds.loc[chns]["gain"].values

    electrons = (adcs - offsets) / gains / PHOTOEL_PER_KEV
    data.insert(0, "ELECTRONS", electrons)
    return data
