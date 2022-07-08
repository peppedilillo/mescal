import argparse
from assets import radsources
from source.io import write_report_to_excel
from source.io import write_report_to_fits
from source.io import write_report_to_csv
from source.errors import FormatNotSupportedError


class SourceNotFoundError(Exception):
    """An error while parsing calib sources."""


parser = argparse.ArgumentParser(description="A script to automatically calibrate HERMES-TP/SP "
                                             "acquisitions of known radioactive sources.")

parser.add_argument("filepath_in",
                    help="input acquisition file in standard 0.5 fits format.")
parser.add_argument("--cache", action="store_true",
                    help="enables loading and saving from cache.")
parser.add_argument("--fmt", default='xslx',
                    help="set output format for calibration and lightoutput tables. "
                         "supported formats: xslx, csv, fits. "
                         "defaults to xslx.")
parser.add_argument("-l", "--lines",
                    help="radioactive sources used for calibration. "
                         "separated by comma, e.g.:  `-l=Fe,Cd,Cs`. "
                         "currently supported sources: Fe, Cd, Cs.")


def get_writer(fmt):
    if fmt == 'xslx':
        return write_report_to_excel
    elif fmt == 'fits':
        return write_report_to_fits
    elif fmt == 'csv':
        return write_report_to_csv
    else:
        raise FormatNotSupportedError("write format not supported")


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

    # TODO: this is shit find better solution
    xlines = {k: v for k, v in sorted(xlines.items(), key=lambda item: item[1])}
    slines = {k: v for k, v in sorted(slines.items(), key=lambda item: item[1])}
    return xlines, slines
