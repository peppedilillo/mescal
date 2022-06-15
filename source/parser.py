import argparse

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
