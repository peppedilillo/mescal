![Alt Text](mescal.gif)

# mescal

Hello and welcome to mescal, your favourite software to analyze HERMES-TP/SP data :).

The packages `lmfit`, `joblib` and `rich`, between others, are required to use mescal.
We provide an `environment.yml` file to easily set up a conda environment in which you can run mescal.
To create the environment from the `environment.yml` file, move to mescal's folder from your terminal and execute:

`conda env create -f environment.yml`

Don't forget to activate the environment afterwards:

`conda activate mescal`

The command line interface of mescal is best rendered on modern terminal applications. 
We are redistributing two great modules by petereon@github, [beaupy](https://github.com/petereon/beaupy) and [yakh](https://github.com/petereon/yakh). 
All hail petereon and all rights reserved.

If you are working on windows, we suggest using mescal with the [new Windows terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701)!


The main goal of mescal is to calibrate the SDDs and scintillators of the HERMES's detector. 

_Example:_
We suppose you have an acquisition of supported radioactive sources ready in the standard HERMES 0.5 fits format.
To launch `mescal.py` over a calibration acquisition (FM1, radioactive sources 55Fe, 109Cd and 137Cs) located at path  "C:\somepath\sources_m20deg_85_LV0d5.fits":

1. Move to the mescal directory.
2. Launch `python mescal.py fm1 Fe,Cd,Cs "C:\somepath\sources_m20deg_85_LV0d5.fits""`

Mescal can also be used to visualize data acquired with the HERMES detector.

For more informations and options regarding `mescal.py`, try our helper via `python mescal.py --help`.
