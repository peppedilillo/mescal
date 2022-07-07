# mescal

Hello and welcome to mescal, your favourite software to analyze HERMES-TP/SP data :).

The packages `lmfit`, `joblib` and `rich` are required to use mescal.
Mescal CLI is best rendered on modern terminal applications. If you are on windows, try it with the [new Windows terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701)!


## calibrate.py

This script is for calibrating the SDD and scintillators of the HERMES detector. 
It supposes you have an acquisition of some radioactive sources ready in the standard HERMES 0.5 fits format.

_Example:_

To launch `calibrate.py` over a calibration acquisition of 55Fe, 109Cd and 137Cs with  "C:\somepath\acquisition_LV0d5.fits":

1. Move to the mescal directory.
2. Launch `python calibrate.py --lines=Fe,Cd,Cs "C:\somepath\acquisition_LV0d5.fits"`

For more informations and options regarding `calibrate.py`, try the helper via `python calibrate.py --help`.
