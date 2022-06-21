'''
this module defines all the paths used across the project.
(so you don't have to go mad in case you need to rethink you output locations)
paths here defined will work across different os.
'''

from pathlib import Path

prjpath = Path(__file__).parent.parent

ASTDIR = prjpath.joinpath("assets")


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
def SPEDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("spectra")


@create_if_not_exists
def DNGDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("diagnostics")


@create_if_not_exists
def FLGDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("flagged")

@create_if_not_exists
def UNCDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("uncalibrated")


LOGOPATH = ASTDIR.joinpath("logo.txt")

EVLFITS = (lambda filepath: RESDIR(filepath).joinpath("event_list.fits"))

FITREPORT = (lambda filepath: RESDIR(filepath).joinpath("report_fit.xlsx"))
CALREPORT = (lambda filepath: RESDIR(filepath).joinpath("report_cal.xlsx"))
SLOREPORT = (lambda filepath: RESDIR(filepath).joinpath("report_slo.xlsx"))

XSPPLOT = (lambda filepath: PLTDIR(filepath).joinpath("spectrum_x.png"))
SSPPLOT = (lambda filepath: PLTDIR(filepath).joinpath("spectrum_s.png"))
QLKPLOT = (lambda filepath: (lambda asic: PLTDIR(filepath).joinpath("quicklook_quad{}.png".format(asic))))
SLOPLOT = (lambda filepath: (lambda asic: PLTDIR(filepath).joinpath("slo_quad{}.png".format(asic))))
LINPLOT = (lambda filepath: (lambda asic, ch: LINDIR(filepath).joinpath("linearity_quad{}_ch{:02d}.png".format(asic, ch))))
SPCPLOT = (lambda filepath: (lambda asic, ch: SPEDIR(filepath).joinpath("spectra_quad{}_ch{:02d}.png".format(asic, ch))))
DNGPLOT = (lambda filepath: (lambda asic, ch: DNGDIR(filepath).joinpath("diagnostic_quad{}_ch{:02d}.png".format(asic, ch))))
FLGPLOT = (lambda filepath: (lambda asic, ch: FLGDIR(filepath).joinpath("flagged_quad{}_ch{:02d}.png".format(asic, ch))))
UNCPLOT = (lambda filepath: (lambda asic, ch: UNCDIR(filepath).joinpath("uncalibrated_quad{}_ch{:02d}.png".format(asic, ch))))
