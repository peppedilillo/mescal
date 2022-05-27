#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Campana, R.; Ceraudo, F.; Della Casa, G.; Dilillo, G.; Marchesini, E. J.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import islice
from astropy.io import fits
from pathlib import Path

import specutilities
import config


def write_(asic, results, to_outputfile):
    to_outputfile.writelines(["{0:s}\t{1:02d}\t".format(asic, v) +
                              "".join(["{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t".format(*fr) for fr in result]) +
                              "\n" for v, result in results.items()])

def channel_fit_plot(x, y, limit, x_fines, fitting_curves, title):
    fig, ax = plt.subplots()
    for l, xf, fc in zip(limit, x_fines, fitting_curves):
        ax.plot(xf, fc, color='red')
        ax.axvspan(l[0], l[1], alpha=0.2, color='green')
    ax.plot(x, y, drawstyle='steps-mid', label='Original data')
    ax.set_title(title)
    return fig


"""
To run this script, a file named calparams_X_*.txt, * being A, B, C or D (quadrant) should be provided
which should include:
1. a line with the number of channels to be analysed, including those that were turned off (i.e., 0, 1, 2, ..., 31)
2. a line with the channels that were turned off
3. a line with the threshold below which all data are discarded (pedestal) for each channel, including those turned off
"""

# visualize=True to visualize every intermediate step through several plots
visualize = True

# smooth=True to apply an smoothing filter to the raw data in order to easily find the peaks
# Note that every other operation, including calibration, is always done with the *non* smoothed data
smooth = True


# ranges[1,2] defines the lower and upper limits of the spectra, in ADC channels
# while step defines how many channels will be contained inside one single bin
ranges = [15000,24000]
step = 7

# if the data are well behaved, they should present only one noise peak/pedestal. if this is the case, then automatic=True
# will look for these values and discard them.
# if instead there are more than one noise peaks/pedestals (for any reason), then the code does not discard them all
# properly. In this case it is better to set automatic=False, and then write down all the minimum ADC values below which the data
# will be discarded in the file _X.txt.
automatic=True

# calibration defines if the calibration is going to be performed or not
calibration = True

# the clean gamma function looks for scintillator-generated events and either keeps only them, or discards them
# if there are no gamma events in your datafile, you can skip this part (cleangamma=False)
cleangamma = True


# threshold defines the threshold above which events are going to be read by the "clean gamma" function
# (meaning: x is considered a gamma event if channel A > 10 AND channel B > 10, A,B scintillator couple)
threshold=10

# keep commands if clean gamma is going to *keep* the gamma rays (and thus discard the X-rays) (keep=True)
# or if clean gamma is going to *discard* the gamma rays (and thus keep the X-rays) (keep=False)
# generally speaking, cal_Xmode operates with keep=False
keep=False

asics = ['A','B','C','D']


fitsfiles = '20220525_125447.fits'
datafile = 'calib_lines.txt'

for i, ASIC in enumerate(asics):
    Path(config.OUTDIR + "/Quad" + ASIC).mkdir(parents=True, exist_ok=True)

    # Files should be named in an uniform way
    # so that the filename depends only on the ASIC
    # fitstiles= [['DatafileX_'+ASIC]]

    # Reading the data from the stretcher_X_*.txt file and slicing the inputs into variables
    with open("stretcher_X_%s.txt" % ASIC, 'r') as f:
        N_channel = np.loadtxt(islice(f, 1), delimiter=" ", dtype=int)
        Off_CHst = np.loadtxt(islice(f, 1), delimiter=" ", dtype=int)
        pedestal = np.loadtxt(islice(f, 1), delimiter=" ", dtype=float)
        couples = np.loadtxt(f, delimiter=" ", dtype=int)
        # on channels
        CHst = np.setdiff1d(range(0, np.size(N_channel)), Off_CHst, assume_unique=False)


    print("Reading file", fitsfile, "using", datafile, "\n")

    # Preparing the variables needed to work
    outputfile_path = config.OUTDIR + '/Quad'+ASIC+'/Quad{:s}_{:s}'.format(ASIC, datafile)
    outputfile, counts_data, line_data, calib_units = specutilities.dataprep(outputfile_path, datafile, fitsfile, ASIC)

    # Cleaning the data by selecting gamma events
    if cleangamma:
        counts_data = specutilities.clean_gamma(counts_data, couples, threshold,keep)

    n_peaks = len(line_data)
    
    
    

    fit_results = {key: np.array([]) for key in CHst}
    fwhm = np.empty([len(N_channel), n_peaks, 2])
    xadc = np.empty([len(N_channel), n_peaks, 2])


    # Fit the defined peaks for the spectrum, one channel at a time
    for v in np.arange(32):
        if v in CHst:  # If the channel is in the list
            window = 5 * step
            w_size = 9

            print('REDUCING ASIC {:s} CH {:02d} STRETCHER\n'.format(ASIC, v))
            channel_data = counts_data.field("CH_{:02d}".format(v))

            # building the raw spectrum
            # the pedestal will be discarded automatically only if automatic=True
            # otherwise remember to set proper threshold values in file calparams_X_*.txt
            x, y = specutilities.hist(pedestal[v], ranges, step, channel_data, 'stretcher')

            plt.figure()
            plt.plot(x,y,drawstyle='steps-mid',label='Original data')
            plt.title("CH {:02d}".format(v))

            try:
                # smoothing (if required), and find the position of each peak
                limit, peak, indexlist = specutilities.detectPeaks(x, y, w_size, n_peaks, window, smooth)
                # Fitting peaks for the given spectrum
                fr, x_fines, fitting_curves = specutilities.fitPeaks(x, y, limit)
                fit_results[v] = fr
            except:
                limit, x_fines, fitting_curves = [], [], []
                fr = np.column_stack(([0.] * n_peaks, [0.] * n_peaks, [0.] * n_peaks, [0.] * n_peaks))
                fit_results[v] = fr

            if visualize:
                channel_fit_plot(x, y, limit, x_fines, fitting_curves, title="CH {:02d}".format(v)).show()

            for p in range(len(fr)):
                fwhm[v, p, 0] = fr[p, 2]
                fwhm[v, p, 1] = fr[p, 3]

                xadc[v, p, 0] = fr[p, 0]
                xadc[v, p, 1] = fr[p, 1]
        else:
            print('Skipping CH {:02d} ASIC {:s} STRETCHER\n'.format(v, ASIC))

    write_(ASIC, fit_results, outputfile)
    outputfile.close()

    if calibration:

        filecal = open(config.OUTDIR + '/Quad' + ASIC + '/Calibration_Stretcher_ASIC_{:s}.txt'.format(ASIC), "w")
        filecal.write("# CH\tGain\tGain_err\tOffset\tOffset_err\tChi2\n")

        gainv = np.empty([len(N_channel), 2])
        offsetv = np.empty([len(N_channel), 2])
        ev_calib = np.empty([len(channel_data), len(N_channel)])

        for v in np.arange(32):
            if v in CHst:  # If the channel is in the list
                # Now calibrate using all the found lines

                print('CALIBRATING ASIC {:s} CH {:02d} STRETCHER\n'.format(ASIC, v))
                channel_data = counts_data.field("CH_{:02d}".format(v))

                chi2, gain, gain_err, offset, offset_err, xcalib_line = specutilities.calibrate(ASIC, v, line_data,
                                                                                  fit_results[v], verbose=False)
                channel_data_calib = (channel_data - offset) / gain

                ev_calib[:, v] = channel_data_calib

                print("Calibration for ASIC {:s} CH {:02d}".format(ASIC, v))
                print("Gain: {0:.3f} +- {1:.3f}\nOffset: {2:.3f} +. {3:.3f}\n".format(gain, gain_err, offset, offset_err))

                gainv[v, 0] = gain
                gainv[v, 1] = gain_err
                offsetv[v, 0] = offset
                offsetv[v, 1] = offset_err


                #    fig1, ax = linPlot('stretcher',ranges,v,xadc[v],xcalib_line,gain,offset,calib_units,figsize=(16,6))
                #    fig1.suptitle('QUADRANT '+ASIC)
                #    fig1.savefig('./Quad'+ASIC+'/'+ASIC+'_LINEARITY_CH'+str(v)+'.png')
                #    plt.close(fig1)

                cal_summary_string = "{:02d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(v, gain, gain_err,
                                                                                               offset, offset_err, chi2)
                filecal.write(cal_summary_string)

                pedestalcal = (pedestal[v] - offsetv[v, 0]) / gainv[v, 0]
                rangescal = (ranges - offsetv[v, 0]) / gainv[v, 0]
                asiccal = (channel_data - offsetv[v, 0]) / gainv[v, 0]
                stepcal = step / gainv[v, 0]

                windowcal = 5 * stepcal

                xcalib, ycalib = specutilities.hist(pedestalcal, rangescal, stepcal, asiccal, 'stretcher')

                with sns.plotting_context("talk"):
                    fig3 = plt.figure(figsize=(16, 6))
                    plt.suptitle('QUADRANT ' + ASIC)

                    specutilities.specPlot(xcalib, ycalib, rangescal, v, n_peaks, xcalib_line, windowcal, 'stretcher')

                    plt.savefig(config.OUTDIR + '/Quad' + ASIC + '/' + ASIC + '_XSPEC_CH_' + str(v) + '.png')
                    plt.close(fig3)


        filecal.close()

        specx = ev_calib[ev_calib >= 2.0]

        table_hdu = fits.BinTableHDU.from_columns([fits.Column(name='X_evt', format='D', array=specx)])
        table_hdu.writeto(config.OUTDIR + '/Quad' + ASIC + '/Spectrum_X_' + ASIC + '.fits', overwrite=True)

    with sns.plotting_context("paper"):

        fig2 = plt.figure(figsize=(16, 6))
        plt.suptitle('QUADRANT ' + ASIC)

        gmean, omean = specutilities.gainPlot(ASIC, N_channel, 'stretcher', gainv, offsetv, fwhm, CHst, Off_CHst)

        plt.savefig(config.OUTDIR + '/Quad' + ASIC + '/' + ASIC + '_GAIN_OFFSET.png')
        plt.close(fig2)
