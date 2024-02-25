# fmt: off
'''
this module defines all the paths used across the project.
(so you don't have to go mad in case you need to rethink you output locations)
paths here defined will work across different os.
'''

from pathlib import Path

prjpath = Path(__file__).parent.parent

VERPATH = prjpath.joinpath("version.txt")
CONFIGPATH = prjpath.joinpath("config.ini")
ASSETDIR = prjpath.joinpath("assets")
LOGOPATH = ASSETDIR.joinpath("logo.txt")


def create_if_not_exists(func):
    def wrapper(*args):
        p = func(*args)
        p.mkdir(exist_ok=True)
        return p
    return wrapper


@create_if_not_exists
def CACHEDIR():
    return prjpath.joinpath("cache")


@create_if_not_exists
def OUTDIR():
    return prjpath.joinpath("output")


@create_if_not_exists
def RESDIR(filepath: Path) -> Path:
    return OUTDIR().joinpath(filepath.name.rstrip(''.join(filepath.suffixes)))


@create_if_not_exists
def PLTDIR(filepath: Path) -> Path:
    return RESDIR(filepath).joinpath("plots")


@create_if_not_exists
def LINDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("linearity")


@create_if_not_exists
def XCSDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("chnspectra_x")


@create_if_not_exists
def SCSDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("chnspectra_s")


@create_if_not_exists
def XDNDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("diagnostics_x")


@create_if_not_exists
def SDNDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("diagnostics_s")


@create_if_not_exists
def TMSDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("lightcurves")


@create_if_not_exists
def UNCDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("uncalibrated")


EVLFITS = (lambda filepath: RESDIR(filepath).joinpath("event_list.fits"))
LOGFILE = (lambda filepath: RESDIR(filepath).joinpath("log.txt"))

XFTREPORT = (lambda filepath: RESDIR(filepath).joinpath("xfit.xlsx"))
SFTREPORT = (lambda filepath: RESDIR(filepath).joinpath("gammafit.xlsx"))
CALREPORT = (lambda filepath: RESDIR(filepath).joinpath("sdds.xlsx"))
RESREPORT = (lambda filepath: RESDIR(filepath).joinpath("resolution.xlsx"))
ELOREPORT = (lambda filepath: RESDIR(filepath).joinpath("lightoutput.xlsx"))
SLOREPORT = (lambda filepath: RESDIR(filepath).joinpath("scintillators.xlsx"))

XSPPLOT = (lambda filepath: PLTDIR(filepath).joinpath("spectrum_x.png"))
SSPPLOT = (lambda filepath: PLTDIR(filepath).joinpath("spectrum_s.png"))
RESPLOT = (lambda filepath: PLTDIR(filepath).joinpath("map_resolution.png"))
CNTPLOT = (lambda filepath: PLTDIR(filepath).joinpath("map_counts.png"))
QLKPLOT = (lambda filepath: (lambda quad: PLTDIR(filepath).joinpath("quicklook_quad{}.png".format(quad))))
SLOPLOT = (lambda filepath: (lambda quad: PLTDIR(filepath).joinpath("slo_quad{}.png".format(quad))))
LINPLOT = (lambda filepath: (lambda quad, ch: LINDIR(filepath).joinpath("linearity_quad{}_ch{:02d}.png".format(quad, ch))))
XCSPLOT = (lambda filepath: (lambda quad, ch: XCSDIR(filepath).joinpath("spectra_x_quad{}_ch{:02d}.png".format(quad, ch))))
SCSPLOT = (lambda filepath: (lambda quad, ch: SCSDIR(filepath).joinpath("spectra_s_quad{}_ch{:02d}.png".format(quad, ch))))
XDNPLOT = (lambda filepath: (lambda quad, ch: XDNDIR(filepath).joinpath("diagnostic_x_quad{}_ch{:02d}.png".format(quad, ch))))
SDNPLOT = (lambda filepath: (lambda quad, ch: SDNDIR(filepath).joinpath("diagnostic_s_quad{}_ch{:02d}.png".format(quad, ch))))
UNCPLOT = (lambda filepath: (lambda quad, ch: UNCDIR(filepath).joinpath("uncalibrated_quad{}_ch{:02d}.png".format(quad, ch))))
TMSPLOT = (lambda filepath: (lambda quad, ch: TMSDIR(filepath).joinpath("lightcurve_quad{}_ch{:02d}.png".format(quad, ch))))
