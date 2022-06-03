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


def create_if_not_exists(func):
    def wrapper(*args):
        p = func(*args)
        p.mkdir(exist_ok=True)
        return p
    return wrapper


@create_if_not_exists
def RESFOLDER(filepath):
    return OUTDIR.joinpath(filepath.name)


@create_if_not_exists
def PLOTFOLDER(filepath):
    return RESFOLDER(filepath).joinpath("plots")


FITREPORT = (lambda filepath, asic: RESFOLDER(filepath).joinpath("fit_report_quad{}.csv".format(asic)))
CALREPORT = (lambda filepath, asic: RESFOLDER(filepath).joinpath("cal_report_quad{}.csv".format(asic)))