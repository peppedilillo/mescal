'''
this module defines all the paths used across the project.
(so you don't have to go mad in case you need to rethink you output locations)
paths here defined will work across different os.
'''

from pathlib import Path

prjpath = Path(__file__).parent.parent

CONFDIR = prjpath.joinpath("configs")
DATADIR = prjpath.joinpath("data")
OUTDIR = prjpath.joinpath("output")
CACHEDIR = prjpath.joinpath("cache")

OUTFILEDIR = (lambda filename: OUTDIR.joinpath(filename))
REPORTS = (lambda filename: OUTFILEDIR().joinpath(filename))


ASICDIR = (lambda filename, asic: OUTDIR.joinpath("Quad{}".format(asic)))
FITREPORT = (lambda filename, asic: ASICDIR(asic).joinpath('Quad{}_fitreport.txt'.format(asic)))
LINPLOT = (lambda asic, v: ASICDIR(asic).joinpath('{}_FITLIN_CH_{:02d}.png'.format(asic, v)))
SPECPLOT = (lambda asic, v: ASICDIR(asic).joinpath(asic + '_XSPEC_CH_{}.png'.format(v)))
XSPECFIT = (lambda asic: ASICDIR(asic).joinpath('Spectrum_X_{}.fits'.format(asic)))
GAINPLOT = (lambda asic: ASICDIR(asic).joinpath('{}_GAIN_OFFSET.png'.format(asic)))
CALIBREPORT = (lambda asic: ASICDIR(asic).joinpath('Calibration_Stretcher_ASIC_{:s}.txt'.format(asic)))