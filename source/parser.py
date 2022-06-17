import argparse
from assets import lines


parser = argparse.ArgumentParser(description="A toolkit for the analysis and visualization of the data"
                                             "produced by the HERMES spacecrafts detectors.")

parser.add_argument("filepath_in",
                    help="input file")
parser.add_argument("--nocache", action="store_true",
                    help="disable loading and saving from cache.")
parser.add_argument("-l", "--lines",
                    help="radioactive sources used for calibration."
                         "provide with no space, e.g.:  `-l FeCdCs`."
                         "currently supported sources: Fe, Cd, Cs, Am.")


def _parse_to_list(source_string):
    if source_string:
        return [source_string[i:i + 2] for i in range(0, len(source_string), 2)]
    return []


def compile_sources_dicts(sources_string):
    xlines = {}
    slines = {}
    sources_list = _parse_to_list(sources_string)
    for element in sources_list:
        if element in lines.x_sources:
            xlines.update(lines.x_sources[element])
        elif element in lines.s_sources:
            slines.update(lines.s_sources[element])
        else:
            raise SourceNotFoundError("unknown calibration source source.")
    return xlines, slines