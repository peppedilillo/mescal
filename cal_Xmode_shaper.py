#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Campana, R.; Ceraudo, F.; Della Casa, G.; Dilillo, G.; Marchesini, E. J.
"""

from itertools import islice
from specutilities import *
from pathlib import Path

# visualize=True to visualize every intermediate step through several plots
visualize = True

# smooth=True to apply an smoothing filter to the raw data in order to easily find the peaks
# Note that every other operation, including calibration, is always done with the *non* smoothed data
smooth = True

# ranges[1,2] defines the lower and upper limits of the spectra, in ADC channels
# while step defines how many channels will be contained inside one single bin
ranges = [0,200]
step = 2

calibration = True

asics = ['A']
fitsfiles = [['QuadA_Shaper.fits']]
datafiles = [['shaper_data.txt']]



for i in range(len(asics)):
    ASIC = asics[i]
    fitsfile_asic = fitsfiles[i]
    datafile_asic = datafiles[i]
    channel, baseline=np.loadtxt('Baseline_'+ASIC+'.txt',dtype=float,unpack=True)

    
    # Reading the data from the couples_*.txt file and slicing the inputs into variables
    with open("couples_%s.txt" % ASIC, 'r') as f:
        N_channel = np.loadtxt(islice(f, 1), delimiter=" ", dtype= int)
    
        OffCH_st = np.loadtxt(islice(f, 1), delimiter=" ", dtype= int)
        CHst = np.setdiff1d(range(0, np.size(N_channel)), OffCH_st, assume_unique=False)

        pedestal = np.loadtxt(islice(f, 1), delimiter=" ", dtype= int)
        
    asic_fit_results = [None]*32
    for j in range(len(fitsfile_asic)):
        fitsfile = fitsfile_asic[j] # Spectrum file
        datafile = datafile_asic[j] # Line info file
        
        print("Calibrating file", fitsfile, "using", datafile, "\n")
    
        # Preparing the variables needed to work
        outputfile, counts_data, line_data = dataprep(datafile, fitsfile, ASIC)
        n_peaks = len(line_data)
        
        Path("./Quad" + ASIC).mkdir(parents=True, exist_ok=True)
    
        # Fit the defined peaks for the spectrum, one channel at a time
        for v in np.arange(32):    
            if v in CHst: # If the channel is in the list
                window = 3*step        # TODO: document! Move to general parameters?
                w_size = 9             # TODO: document! Move to general parameters?
    
                print('REDUCING ASIC {:s} CH {:02d} SHAPER\n'.format(ASIC, v))
              
                channel_data = counts_data.field("CH_{:02d}".format(v))
                channel_data = channel_data + baseline[v]-baseline[0]


                # building the raw spectrum
                x, y = hist(pedestal[v], ranges, step, channel_data, 'shaper')
            
                if visualize:
                    plt.figure()
                    plt.plot(x,y,drawstyle='steps-mid')
                    plt.title("CH {:02d}".format(v))
                
                try:
                    # smoothing (if required), and find the position of each peak
                    limit, peak, indexlist = detectPeaks(x, y, w_size, n_peaks, window, smooth)
                    for l in limit:
                        plt.axvspan(l[0], l[1], alpha=0.2, color='green')

                    # Fitting peaks for the given spectrum
                    fit_results = fitPeaks(x, y, limit, visualize=visualize)
                    if asic_fit_results[v] is not None:
                        asic_fit_results[v] = np.row_stack((asic_fit_results[v], fit_results))
                    else:
                        asic_fit_results[v] = fit_results
                except:
                    fit_results = np.column_stack(([0.]*n_peaks, [0.]*n_peaks, [0.]*n_peaks, [0.]*n_peaks))
                    asic_fit_results[v] = fit_results
                    
                # Add line in summary file
                output_summary_string = "{0:s}\t{1:02d}\t".format(ASIC, v)
                for p in range(len(fit_results)):
                    output_summary_string += "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t".format(fit_results[p,0],fit_results[p,2],fit_results[p,1],fit_results[p,3])
                outputfile.write(output_summary_string+"\n")
            else:
                print ('Skipping CH {:02d} ASIC {:s} SHAPER\n'.format(v, ASIC)) 
        outputfile.close()
        
    if calibration:
        filecal = open("Calibration_Shaper_ASIC_{:s}.txt".format(ASIC), "w")
        filecal.write("# CH\tGain\tGain_err\tOffset\tOffset_err\n")
        for v in np.arange(32):
            if v in CHst: # If the channel is in the list
                # Now calibrate using all the found lines
                chi2, gain, gain_err, offset, offset_err = calibrate(line_data, asic_fit_results[v], verbose=False)
                #salvare tutto in array di 32 canali
                print("Calibration for ASIC {:s} CH {:02d}".format(ASIC, v))
                print("Gain: {0:.3f} +- {1:.3f}\nOffset: {2:.3f} +. {3:.3f}\n".format(gain, gain_err, offset, offset_err))
#1.
# loop che usa il array salvato di gain e offset e chiama a gainPlot
#def gainPlot(asic,ch,mode,gain,offset,fwhm,onchannels,offchannels,window):


#2.
#loop che faccia la conversione in keV del histogram x,y
#con questo histogram x_energy,y devo chiamare la function CalPlot
#def CalPlot(x,y,npeak,lines,dtitle,color,mode):


                cal_summary_string = "{:02d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(v, gain, gain_err, offset, offset_err, chi2)
                filecal.write(cal_summary_string)
        filecal.close()
    
if visualize:      
    plt.show()
