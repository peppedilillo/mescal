'''
this module defines all the paths used across the project.
(so you don't have to go mad in case you need to rethink you output locations)
paths here defined will work across different os.
'''

from pathlib import Path


prjpath = Path(__file__).parent.parent


ASSETDIR = prjpath.joinpath("assets")
LOGOPATH = ASSETDIR.joinpath("logo.txt")
DEFCALDIR = ASSETDIR.joinpath("default_calibrations")
FM1CALDIR = DEFCALDIR.joinpath("fm1")
FM1Tm20DIR = FM1CALDIR.joinpath("20220616_55Fe109Cd137Cs_m20deg_thr105_LV0d5")
FM1Tm10DIR = FM1CALDIR.joinpath("20220623_55Fe109Cd137Cs_m10deg_thr105_LV0d5")
FM1Tp00DIR = FM1CALDIR.joinpath("20220622_55Fe109Cd137Cs_0deg_thr105_LV0d5")
FM1Tp20DIR = FM1CALDIR.joinpath("20220622_55Fe109Cd137Cs_20deg_thr105_LV0d5")

FM1Tm20CAL = FM1Tm20DIR.joinpath("report_cal.xlsx")
FM1Tm20SLO = FM1Tm20DIR.joinpath("report_slo.xlsx")
FM1Tm10CAL = FM1Tm10DIR.joinpath("report_cal.xlsx")
FM1Tm10SLO = FM1Tm10DIR.joinpath("report_slo.xlsx")
FM1Tp00CAL = FM1Tp00DIR.joinpath("report_cal.xlsx")
FM1Tp00SLO = FM1Tp00DIR.joinpath("report_slo.xlsx")
FM1Tp20CAL = FM1Tp20DIR.joinpath("report_cal.xlsx")
FM1Tp20SLO = FM1Tp20DIR.joinpath("report_slo.xlsx")


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
def FLGDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("flagged")


@create_if_not_exists
def UNCDIR(filepath: Path) -> Path:
    return PLTDIR(filepath).joinpath("uncalibrated")


EVLFITS = (lambda filepath: RESDIR(filepath).joinpath("event_list.fits"))

XFTREPORT = (lambda filepath: RESDIR(filepath).joinpath("report_xfit.xlsx"))
SFTREPORT = (lambda filepath: RESDIR(filepath).joinpath("report_sfit.xlsx"))
CALREPORT = (lambda filepath: RESDIR(filepath).joinpath("report_cal.xlsx"))
SLOREPORT = (lambda filepath: RESDIR(filepath).joinpath("report_slo.xlsx"))

XSPPLOT = (lambda filepath: PLTDIR(filepath).joinpath("spectrum_x.png"))
SSPPLOT = (lambda filepath: PLTDIR(filepath).joinpath("spectrum_s.png"))
QLKPLOT = (lambda filepath: (lambda quad: PLTDIR(filepath).joinpath("quicklook_quad{}.png".format(quad))))
SLOPLOT = (lambda filepath: (lambda quad: PLTDIR(filepath).joinpath("slo_quad{}.png".format(quad))))
LINPLOT = (lambda filepath: (lambda quad, ch: LINDIR(filepath).joinpath("linearity_quad{}_ch{:02d}.png".format(quad, ch))))
XCSPLOT = (lambda filepath: (lambda quad, ch: XCSDIR(filepath).joinpath("spectra_x_quad{}_ch{:02d}.png".format(quad, ch))))
SCSPLOT = (lambda filepath: (lambda quad, ch: SCSDIR(filepath).joinpath("spectra_s_quad{}_ch{:02d}.png".format(quad, ch))))
XDNPLOT = (lambda filepath: (lambda quad, ch: XDNDIR(filepath).joinpath("diagnostic_x_quad{}_ch{:02d}.png".format(quad, ch))))
SDNPLOT = (lambda filepath: (lambda quad, ch: SDNDIR(filepath).joinpath("diagnostic_s_quad{}_ch{:02d}.png".format(quad, ch))))
FLGPLOT = (lambda filepath: (lambda quad, ch: FLGDIR(filepath).joinpath("flagged_quad{}_ch{:02d}.png".format(quad, ch))))
UNCPLOT = (lambda filepath: (lambda quad, ch: UNCDIR(filepath).joinpath("uncalibrated_quad{}_ch{:02d}.png".format(quad, ch))))
