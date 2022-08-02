import numpy as np
import pandas as pd

LO_PARAMS = [
    "light_out",
    "light_out_err",
]


def as_dict_of_dataframes(f):
    def wrapper(*args):
        nested_dict = f(*args)
        quadrants = nested_dict.keys()

        dict_of_dfs = {
            q: pd.DataFrame(
                nested_dict[q],
                index=LO_PARAMS
            ).T.rename_axis("channel")
            for q in quadrants
        }
        return dict_of_dfs
    return wrapper


@as_dict_of_dataframes
def compute_los(sfit_results, scint_calibs, couples):
    results = {}
    for quad in sfit_results.keys():
        sfit = sfit_results[quad]
        scint = scint_calibs[quad]
        quad_couples = dict(couples[quad])
        inverted_quad_couples = {v: k for k, v in quad_couples.items()}
        for ch in sfit.index:
            try:
                companion = quad_couples[ch]
            except KeyError:
                companion = inverted_quad_couples[ch]

            if ch in scint.index:
                lo = scint.loc[ch]['light_out']
                lo_err = scint.loc[ch]['light_out_err']
            else:
                lo = scint.loc[companion]['light_out']
                lo_err = scint.loc[companion]['light_out_err']

            centers = sfit.loc[ch][:,'center'].values
            centers_companion = sfit.loc[companion][:,'center'].values
            effs = lo*centers/(centers+centers_companion)
            eff_errs = lo_err*centers/(centers+centers_companion)
            eff, eff_err = deal_with_multiple_gamma_decays(effs, eff_errs)
            results.setdefault(quad, {})[ch] = np.array((eff, eff_err))
    return results



@as_dict_of_dataframes
def calibrate_scintillators(sumfit_results, radsources):
    energies = [s.energy for s in radsources.values()]

    results = {}
    for quad in sumfit_results.keys():
        df = sumfit_results[quad]
        for ch in df.index:
            los = df.loc[ch][:,'center'].values/energies
            lo_errs = df.loc[ch][:,'center_err'].values/energies

            lo, lo_err = deal_with_multiple_gamma_decays(los, lo_errs)
            results.setdefault(quad, {})[ch] = np.array((lo, lo_err))
    return results


def deal_with_multiple_gamma_decays(light_outs, light_outs_errs):
    # in doubt mean
    return light_outs.mean(), np.sqrt(np.sum(light_outs_errs ** 2))