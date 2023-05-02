# fmt: off
from collections import namedtuple

from source.errors import SourceNotFoundError

Decay = namedtuple('Decay', ['energy', 'low_lim', 'hi_lim'])

_55fe = {
    'Fe 5.9 keV': Decay(5.90, -2., +1.),
}

_109cd = {
    'Cd 22.1 keV': Decay(22.16, -1., + 2.),
    'Cd 24.9 keV': Decay(24.94, -2., + 1.),
}

_241am = {
    'Am 13.9 keV': Decay(13.9, -2., +2.),
    'Am 17.7 keV': Decay(17.7, -2., +2.),
    'Am 20.7 keV': Decay(20.7, -2., +2.),
    'Am 26.3 keV': Decay(26.3, -2., +2.),
}

_241am_x60 = {
    'Am 13.9 keV': Decay(13.9, -2., +2.),
    'Am 17.7 keV': Decay(17.7, -2., +2.),
    'Am 20.7 keV': Decay(20.7, -2., +2.),
    'Am 26.3 keV': Decay(26.3, -2., +2.),
    'Am 59.5 keV': Decay(59.5, -2., +2.),
}

_137cs = {
    'Cs 662 keV': Decay(661.6, -1., +2.),
}

_xsources = {
    "55Fe": _55fe,
    "109Cd": _109cd,
}

_ssources = {
    "137Cs": _137cs,
}


def supported_sources():
    return list(_xsources.keys()) + list(_ssources.keys())


def radsources_dicts(sources: list):
    xdecays = {}
    sdecays = {}
    for element in sources:
        if element in _xsources:
            xdecays.update(_xsources[element])
        elif element in _ssources:
            sdecays.update(_ssources[element])

    xdecays = {
        k: v for k, v in sorted(xdecays.items(), key=lambda item: item[1])
    }
    sdecays = {
        k: v for k, v in sorted(sdecays.items(), key=lambda item: item[1])
    }
    return xdecays, sdecays
