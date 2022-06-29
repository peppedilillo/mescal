import argparse
from assets import radsources


class SourceNotFoundError(Exception):
    """An error while parsing calib sources."""


parser = argparse.ArgumentParser(description="A toolkit for the analysis and visualization of the data"
                                             "produced by the HERMES spacecrafts detectors.")

parser.add_argument("filepath_in",
                    help="input file")
parser.add_argument("--cache", action="store_true",
                    help="enables loading and saving from cache.")
parser.add_argument("-l", "--lines",
                    help="radioactive sources used for calibration."
                         "separated by comma, e.g.:  `-l=Fe,Cd,Cs`."
                         "currently supported sources: Fe, Cd, Cs, Am.")


def _parse_to_list(source_string):
    if source_string:
        return source_string.upper().split(",")
    return []


def compile_sources_dicts(sources_string):
    xlines = {}
    slines = {}
    sources_list = _parse_to_list(sources_string)
    for element in sources_list:
        if element in radsources.x_sources:
            xlines.update(radsources.x_sources[element])
        elif element in radsources.s_sources:
            slines.update(radsources.s_sources[element])
        else:
            raise SourceNotFoundError("unknown calibration source source.")
    return xlines, slines
