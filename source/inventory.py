import numpy as np

from assets import detectors
from assets.radsources_db import Fe
from assets.radsources_db import Fe_kbeta
from assets.radsources_db import Cd
from assets.radsources_db import Am
from assets.radsources_db import Am_x60
from assets.radsources_db import Cs
from source.upaths import FM1Tm20CAL
from source.upaths import FM1Tm20SLO
from source.upaths import FM1Tm10CAL
from source.upaths import FM1Tm10SLO
from source.upaths import FM1Tp00CAL
from source.upaths import FM1Tp00SLO
from source.upaths import FM1Tp20CAL
from source.upaths import FM1Tp20SLO
from source.errors import DetectorModelNotFound
from source.errors import SourceNotFoundError
from source.io import read_report_from_excel

X_SOURCES = {
    'FE': Fe,
    'FE_KBETA': Fe_kbeta,
    'CD': Cd,
    'AM': Am,
    'AM_X60': Am_x60,
}

GAMMA_SOURCES = {
    'CS': Cs,
}

SDD_CALIBS = {
    ('fm1', -20): FM1Tm20CAL,
    ('fm1', -10): FM1Tm10CAL,
    ('fm1', 0):   FM1Tp00CAL,
    ('fm1', +20): FM1Tp20CAL,
}


SLO_CALIBS = {
    ('fm1', -20): FM1Tm20SLO,
    ('fm1', -10): FM1Tm10SLO,
    ('fm1', 0):   FM1Tp00SLO,
    ('fm1', +20): FM1Tp20SLO,
}

ROOM_TEMP = +20

def available_temps(model, calibs):
    out = [temp
           for available_model, temp in calibs.keys()
           if model == available_model]
    return out


def available_models(calibs):
    models, _ = zip(*calibs.keys())
    return [*set(models)]


def fetch_default_sdd_calibration(model, temp):
    if model in available_models(SDD_CALIBS) and (temp is not None):
        nearest_available_temperature = min(available_temps(model, SDD_CALIBS)[::-1],
                                            key=lambda x: abs(x - temp))
        calibration_path = SDD_CALIBS[(model, nearest_available_temperature)]
        calibration_df = read_report_from_excel(calibration_path)
        return calibration_df, (model, nearest_available_temperature)

    if model in available_models(SDD_CALIBS) and (temp is None):
        calibration_path = SDD_CALIBS[(model, ROOM_TEMP)]
        calibration_df = read_report_from_excel(calibration_path)
        return calibration_df, (model, ROOM_TEMP)

    else:
        raise DetectorModelNotFound("model not available.")


def fetch_default_slo_calibration(model, temp):
    if model in available_models(SLO_CALIBS) and temp in available_temps(model, SLO_CALIBS):
        nearest_available_temperature = min(available_temps(model, SLO_CALIBS)[::-1],
                                            key=lambda x: abs(x - temp))
        calibration_path = SLO_CALIBS[(model, nearest_available_temperature)]
        calibration_df = read_report_from_excel(calibration_path)
        return calibration_df, (model, nearest_available_temperature)

    if model in available_models(SLO_CALIBS) and (temp is None):
        calibration_path = SLO_CALIBS[(model, ROOM_TEMP)]
        calibration_df = read_report_from_excel(calibration_path)
        return calibration_df, (model, ROOM_TEMP)

    else:
        raise DetectorModelNotFound("model not available.")


def get_quadrant_map(quad: str, arr_borders: bool = True):
    if quad in ['A', 'B', 'C', 'D']:
        arr = detectors.map[quad]
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


def get_quad_couples(quad):
    qmaparr = np.array(get_quadrant_map(quad))
    arr = np.lexsort((qmaparr[:, 0], qmaparr[:, 1])).reshape(16, 2)[1:]
    dic = dict(arr)
    return dic


def get_couples():
    return {q: get_quad_couples(q) for q in "ABCD"}
