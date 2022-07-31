import numpy as np
import pandas as pd
from lmfit.models import LinearModel
from source.errors import FailedFitError
from source.errors import warn_failed_linearity_fit
import logging

CAL_PARAMS = [
    "gain",
    "gain_err",
    "offset",
    "offset_err",
    "chi2",
]


def as_dict_of_dataframes(f):
    def wrapper(*args):
        nested_dict, radsources, *etc = f(*args)
        quadrants = nested_dict.keys()

        dict_of_dfs = {
            q: pd.DataFrame(
                nested_dict[q],
                index=CAL_PARAMS
            ).T.rename_axis("channel")
            for q in quadrants
        }
        return dict_of_dfs, *etc
    return wrapper


@as_dict_of_dataframes
def calibrate_sdds(radsources, fits):
    results, flagged = {}, {}

    for quad in fits.keys():
        for ch in fits[quad].index:
            labels = fits[quad].loc[ch].index.unique(level='source').to_list()
            energies = [radsources[label].energy for label in labels]
            centers = fits[quad].loc[ch][:, 'center'].values
            center_errs = fits[quad].loc[ch][:, 'center_err'].values

            try:
                cal_results = _calibrate_chn(
                    centers,
                    energies,
                    weights=center_errs,
                )
            except FailedFitError:
                message = warn_failed_linearity_fit(quad, ch)
                logging.warning(message)
                flagged.setdefault(quad, []).append(ch)
                continue

            results.setdefault(quad, {})[ch] = np.array(cal_results)
    return results, radsources, flagged


def _calibrate_chn(centers, radsources: list, weights=None):
    lmod = LinearModel()
    pars = lmod.guess(centers, x=radsources)
    try:
        resultlin = lmod.fit(centers, pars, x=radsources, weights=weights)
    except ValueError:
        raise FailedFitError("linear fitter error")

    chi2 = resultlin.redchi
    gain = resultlin.params['slope'].value
    offset = resultlin.params['intercept'].value
    gain_err = resultlin.params['slope'].stderr
    offset_err = resultlin.params['intercept'].stderr

    return gain, gain_err, offset, offset_err, chi2