# Hermes-Cal-SW
Hermes TP/SP Calibration Software Database

The Hermes TP/SP Calibration Software currently consists of three main blocks. In this repository, we update each block separately within three different branches. After agreeing upon a new version of one of these blocks, its corresponding branch will be merged into the main code, thus creating a new version of the SW. *These* versions will be numbered; as of 19/05/2022 we are working with the v0.0 version.

The three main blocks (and branches) are:

1) X mode:

This code aims to perform a energy calibration on the X mode, i.e. 2-60 keV (upper limit given by the source during calibration measures).
It includes the automated detection of X-ray peaks (Am241, Cd109, Fe55...), the fitting of those peaks, the linear regression between the ADC-keV dimensions, and several plots, including gain and offset quicklook, detector linearity, and calibrated X spectrum.

This code in particular was written in such a way that it can be used also with electric impulses instead of radioactive sources, since the linear regression is automated for all types of measure units.

2) S mode:

This code aims to obtain the light output for each scintillator, and to build a gamma-ray spectrum.
It includes the automated detection of *A* gamma-ray peak (Cs137), the fitting of such peak, and the calibration through the Gain and Offset values obtained in the X mode of each channel, to then normalise it to the light output of each crystal. Finally, spectra are built by summing the events of each coupled channel with its partner.

3) Specutilities

This is a package of several functions that are used throughout the whole code. It includes histogram building, linear regression, gaussian fitting, peak detection, plot building, data preparation, and more. 


