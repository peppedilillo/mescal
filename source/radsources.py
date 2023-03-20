# fmt: off
from collections import namedtuple

from source.errors import SourceNotFoundError

Decay = namedtuple('Decay', ['energy', 'low_lim', 'hi_lim'])

Fe = {
    'Fe 5.9 keV': Decay(5.90, -2., +1.),
}

Fe_kbeta = {
    'Fe 5.9 keV': Decay(5.90, -2., +1.),
    'Fe 6.4 keV': Decay(6.49, -1., +2.),
}

Cd = {
    'Cd 22.1 keV': Decay(22.16, -1., + 2.),
    'Cd 24.9 keV': Decay(24.94, -2., + 1.),
}

Am = {
    'Am 13.9 keV': Decay(13.9, -2., +2.),
    'Am 17.7 keV': Decay(17.7, -2., +2.),
    'Am 20.7 keV': Decay(20.7, -2., +2.),
    'Am 26.3 keV': Decay(26.3, -2., +2.),
}

Am_x60 = {
    'Am 13.9 keV': Decay(13.9, -2., +2.),
    'Am 17.7 keV': Decay(17.7, -2., +2.),
    'Am 20.7 keV': Decay(20.7, -2., +2.),
    'Am 26.3 keV': Decay(26.3, -2., +2.),
    'Am 59.5 keV': Decay(59.5, -2., +2.),
}

Cs = {
    'Cs 662 keV': Decay(661.6, -1., +2.),
}

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

    xdecays = {
        k: v for k, v in sorted(xdecays.items(), key=lambda item: item[1])
    }
    sdecays = {
        k: v for k, v in sorted(sdecays.items(), key=lambda item: item[1])
    }
    return xdecays, sdecays
