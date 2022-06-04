import argparse

parser = argparse.ArgumentParser(description="A toolkit for the analysis and visualization of the data"
                                             "produced by the HERMES spacecrafts detectors.")

parser.add_argument("filepath_in",
                    help="input file")
parser.add_argument("--nocache", action="store_true",
                    help="disable loading and saving from cache.")