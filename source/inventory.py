import numpy as np

import source.paths as paths
from assets import detectors
from assets.radsources_db import Am, Am_x60, Cd, Cs, Fe, Fe_kbeta
from source.errors import DetectorModelNotFound, SourceNotFoundError
from source.io import read_report_from_excel

X_SOURCES = {
    "FE": Fe,
    "FE_KBETA": Fe_kbeta,
    "CD": Cd,
    "AM": Am,
    "AM_X60": Am_x60,
}


GAMMA_SOURCES = {
    "CS": Cs,
}


SDD_CALIBS = {
    ("fm1", -20): paths.FM1Tm20CAL,
    ("fm1", -10): paths.FM1Tm10CAL,
    ("fm1", 0): paths.FM1Tp00CAL,
    ("fm1", +20): paths.FM1Tp20CAL,
    ("pfm", +20): paths.PFMTp20CAL,
    ("pfm", +10): paths.PFMTp10CAL,
    ("pfm", 0): paths.PFMTp00CAL,
    ("pfm", -10): paths.PFMTm10CAL,
    ("pfm", -20): paths.PFMTm20CAL,
}


ROOM_TEMP = +20


def available_temps_hints(model, calibs):
    out = [temp for available_model, temp in calibs.keys() if model == available_model]
    return out


def available_models_hints(calibs):
    models, _ = zip(*calibs.keys())
    return [*set(models)]


def fetch_default_sdd_calibration(model, temp):
    # if a default calibs for the selected detector models exists
    # and a temperature was specified, search for the calibrations at
    # closest temperature
    if model in available_models_hints(SDD_CALIBS) and (temp is not None):
        nearest_available_temperature = min(
            available_temps_hints(model, SDD_CALIBS)[::-1], key=lambda x: abs(x - temp)
        )
        calibration_path = SDD_CALIBS[(model, nearest_available_temperature)]
        calibration_df = read_report_from_excel(calibration_path)
        return calibration_df, (model, nearest_available_temperature)

    # if temperature is not specified use a default value.
    if model in available_models_hints(SDD_CALIBS) and (temp is None):
        calibration_path = SDD_CALIBS[(model, ROOM_TEMP)]
        calibration_df = read_report_from_excel(calibration_path)
        return calibration_df, (model, ROOM_TEMP)

    else:
        raise DetectorModelNotFound("model not available.")


def get_quadrant_map(model: str, quad: str, arr_borders: bool = True):
    if model == "fm1":
        detector_map = detectors.fm1
    elif model == "pfm":
        detector_map = detectors.pfm
    elif model == "fm2":
        detector_map = detectors.fm2
    else:
        raise ValueError("Model Unknown.")

    if quad in ["A", "B", "C", "D"]:
        arr = detector_map[quad]
    else:
        raise ValueError("Unknown quadrant key. Allowed keys are A,B,C,D")

    if arr_borders:
        return tuple(map(lambda x: (x[0] + int(x[0] / 2), x[1]), arr))

    return arr


def radsources_dicts(sources: list):
    xdecays = {}
    sdecays = {}
    for element in sources:
        if element in X_SOURCES:
            xdecays.update(X_SOURCES[element])
        elif element in GAMMA_SOURCES:
            sdecays.update(GAMMA_SOURCES[element])
        else:
            raise SourceNotFoundError("unknown calibration source source.")

    xdecays = {k: v for k, v in sorted(xdecays.items(), key=lambda item: item[1])}
    sdecays = {k: v for k, v in sorted(sdecays.items(), key=lambda item: item[1])}
    return xdecays, sdecays


def get_quad_couples(model, quad):
    qmaparr = np.array(get_quadrant_map(model, quad))
    arr = np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]
    dic = dict(arr)
    return dic


def get_couples(model):
    return {q: get_quad_couples(model, q) for q in "ABCD"}
