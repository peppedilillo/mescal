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
from source.errors import UnknownModelError
from source.errors import CalibrationNotFoundError
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

AVAILABLE_MODELS = [
    'fm1',
]

AVAILABLE_TEMP = {
    'fm1': [-20, -10, 0, 20],
}

AVAILABLE_SDD_CALIBS = {
    ('fm1', -20): FM1Tm20CAL,
    ('fm1', -10): FM1Tm10CAL,
    ('fm1', 0):   FM1Tp00CAL,
    ('fm1', +20): FM1Tp20CAL,
}

AVAILABLE_SLO_CALIBS = {
    ('fm1', -20): FM1Tm20SLO,
    ('fm1', -10): FM1Tm10SLO,
    ('fm1', 0):   FM1Tp00SLO,
    ('fm1', +20): FM1Tp20SLO,
}


def compile_sources_dicts(sources: list):
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


def fetch_default_sdd_calibration(model, temp):
    if model in AVAILABLE_MODELS:
        nearest_available_temperature = min(AVAILABLE_TEMP[model],
                                            key=lambda x: abs(x - temp))
        calibration_path = AVAILABLE_SDD_CALIBS[(model, nearest_available_temperature)]
        calibration_df = read_report_from_excel(calibration_path)
        return calibration_df
    else:
        raise CalibrationNotFoundError("model not available.")


def fetch_default_slo_calibration(model, temp):
    if model in AVAILABLE_MODELS:
        nearest_available_temperature = min(AVAILABLE_TEMP[model],
                                            key=lambda x: abs(x - temp))
        calibration_path = AVAILABLE_SLO_CALIBS[(model, nearest_available_temperature)]
        calibration_df = read_report_from_excel(calibration_path)
        return calibration_df
    else:
        raise CalibrationNotFoundError("model not available.")


def get_quadrant_map(model: str, quad: str, arr_borders: bool = True):
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