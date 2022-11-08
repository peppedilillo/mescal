# fmt: off
from collections import namedtuple

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
