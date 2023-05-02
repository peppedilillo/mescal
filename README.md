# mescal

Hello and welcome to mescal, your favourite software to analyze data from the HERMES-SP payloads.

![mescal's tui](mescal.gif)

## Setup 

The packages `lmfit`, `joblib` and `rich`, between others, are required to use mescal.
We provide an `environment.yml` file to easily set up a conda environment in which you can run mescal.
To create the environment from the `environment.yml` file, move to mescal's folder from your terminal and execute:

`conda env create -f environment.yml`

The command line interface of mescal is best rendered on modern terminal applications. 
We are redistributing two great modules by petereon@github, [beaupy](https://github.com/petereon/beaupy) and [yakh](https://github.com/petereon/yakh). 
All hail petereon and all rights reserved.

If you are working on windows, we suggest using mescal with the [new Windows terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701)!

## Running mescal

The main goal of mescal is to calibrate the SDDs and scintillators of the HERMES's detector. 

To launch `mescal`:

1. Move to the mescal directory.
2. Activate the mescal environment with `conda activate mescal`
3. Launch `python mescal.py`
4. Follow the instructions on screen.

Mescal can also be used to visualize data acquired with the HERMES detector.

## Helper

A few options (e.g., output format, DAC presets..) can be activated via command line arguments. 
You can check these through mescal's helper: `python mescal.py --help`.


