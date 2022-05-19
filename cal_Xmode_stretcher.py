#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Campana, R.; Ceraudo, F.; Della Casa, G.; Dilillo, G.; Marchesini, E. J.
"""

import numpy as np
import lmfit
import re
import seaborn as sns
from lmfit.models import GaussianModel, LinearModel, PolynomialModel, Model
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import os
from itertools import islice
import statistics
from specutilities import *
from pathlib import Path
import warnings

"""
To run this script, a file named stretcher_X_*.txt, * being A, B, C or D (quadrant) should be provided
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
ranges = [1000,2400]
step = 2

calibration = True
cleangamma = False
asics = ['C']

fitsfiles = ['QC_FeAm.fits']
datafile = 'calib_lines.txt'


for i in range(len(asics)):
    ASIC = asics[i]
    

    Path("./Quad" + ASIC).mkdir(parents=True, exist_ok=True)

    #Files should be named in an uniform way
    #so that the filename depends only on the ASIC
    #fitstiles= [['DatafileX_'+ASIC]]
    
    fitsfile = fitsfiles[i]

    # Reading the data from the stretcher_X_*.txt file and slicing the inputs into variables
    with open("stretcher_X_%s.txt" % ASIC, 'r') as f:
        N_channel = np.loadtxt(islice(f, 1), delimiter=" ", dtype= int)
    
        Off_CHst = np.loadtxt(islice(f, 1), delimiter=" ", dtype= int)
        CHst = np.setdiff1d(range(0, np.size(N_channel)), Off_CHst, assume_unique=False)

        pedestal = np.loadtxt(islice(f, 1), delimiter=" ", dtype= float)

        couples = np.loadtxt(f, delimiter=" ", dtype= int)



    asic_fit_results = [None]*32


    print("Reading file", fitsfile, "using", datafile, "\n")
    
    # Preparing the variables needed to work
    outputfile, counts_data, line_data, calib_units = dataprep(datafile, fitsfile, ASIC)


    # Cleaning the data by selecting gamma events
    if cleangamma:

        print(couples)  
        counts_data = clean_gamma(counts_data, couples, threshold=1100)          
        
    
    n_peaks = len(line_data)
    
    
    

    fwhm=np.empty([len(N_channel),n_peaks,2])
    xadc=np.empty([len(N_channel),n_peaks,2])


    # Fit the defined peaks for the spectrum, one channel at a time
    for v in np.arange(32):    
        if v in CHst: # If the channel is in the list
            window = 5*step         
            w_size = 9              
    
            print('REDUCING ASIC {:s} CH {:02d} STRETCHER\n'.format(ASIC, v))
        
            channel_data = counts_data.field("CH_{:02d}".format(v))

            # building the raw spectrum
            x, y = hist(pedestal[v], ranges, step, channel_data, 'stretcher')

            plt.figure()
            plt.plot(x,y,drawstyle='steps-mid',label='Original data')
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
            
            if visualize:
                plt.show()    
            
            # Add line in summary file
            output_summary_string = "{0:s}\t{1:02d}\t".format(ASIC, v)


            for p in range(len(fit_results)):
                output_summary_string += "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t".format(fit_results[p,0],fit_results[p,2],fit_results[p,1],fit_results[p,3])


                fwhm[v,p,0]=fit_results[p,2]
                fwhm[v,p,1]=fit_results[p,3]

                xadc[v,p,0]=fit_results[p,0]
                xadc[v,p,1]=fit_results[p,1]


            outputfile.write(output_summary_string+"\n")
        else:
            print ('Skipping CH {:02d} ASIC {:s} STRETCHER\n'.format(v, ASIC)) 
    outputfile.close()



    if calibration:
        
        filecal = open('./Quad'+ASIC+'/Calibration_Stretcher_ASIC_{:s}.txt'.format(ASIC), "w")
        filecal.write("# CH\tGain\tGain_err\tOffset\tOffset_err\tChi2\n")
        
        gainv=np.empty([len(N_channel),2])
        offsetv=np.empty([len(N_channel),2])
        ev_calib=np.empty([len(channel_data),len(N_channel)])

        
        for v in np.arange(32):
            if v in CHst: # If the channel is in the list
                # Now calibrate using all the found lines
    
                print('CALIBRATING ASIC {:s} CH {:02d} STRETCHER\n'.format(ASIC, v))   
                channel_data = counts_data.field("CH_{:02d}".format(v))     

                chi2, gain, gain_err, offset, offset_err, xcalib_line = calibrate(ASIC,v,line_data, asic_fit_results[v], verbose=False)

                channel_data_calib=(channel_data-offset)/gain
                
                ev_calib[:,v]=channel_data_calib

                print("Calibration for ASIC {:s} CH {:02d}".format(ASIC, v))
                print("Gain: {0:.3f} +- {1:.3f}\nOffset: {2:.3f} +. {3:.3f}\n".format(gain, gain_err, offset, offset_err))

                gainv[v,0]=gain
                gainv[v,1]=gain_err
                offsetv[v,0]=offset
                offsetv[v,1]=offset_err
                

                with sns.plotting_context("talk"):


                    fig1, ax = linPlot('stretcher',ranges,v,xadc[v],xcalib_line,gain,offset,calib_units,figsize=(16,6))
                    fig1.suptitle('QUADRANT '+ASIC)
                    fig1.savefig('./Quad'+ASIC+'/'+ASIC+'_LINEARITY_CH'+str(v)+'.png')
                    plt.close(fig1)


                cal_summary_string = "{:02d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(v, gain, gain_err, offset, offset_err, chi2)
                filecal.write(cal_summary_string)

                pedestalcal=(pedestal[v]-offsetv[v,0])/gainv[v,0]
                rangescal=(ranges-offsetv[v,0])/gainv[v,0]
                asiccal=(channel_data-offsetv[v,0])/gainv[v,0]
                stepcal=step/gainv[v,0]
                
                windowcal=5*stepcal

                xcalib,ycalib=hist(pedestalcal,rangescal,stepcal,asiccal,'stretcher')

                with sns.plotting_context("talk"):
                    fig3=plt.figure(figsize=(16,6))
                    plt.suptitle('QUADRANT '+ASIC)
        
                    specPlot(xcalib,ycalib,rangescal,v,n_peaks,xcalib_line,windowcal,'stretcher')
    
                    plt.savefig('./Quad'+ASIC+'/'+ASIC+'_XSPEC_CH_'+str(v)+'.png')
                    plt.close(fig3)
    
                #hdulistfe = fits.open('QA_Fe55.fits')
                #counts_data_fe = hdulistfe[1].data
                #hdulistfe.close()
    #
                #channel_data_fe = counts_data_fe.field("CH_{:02d}".format(v))     
#
                #asiccal_fe=(channel_data_fe-offsetv[v,0])/gainv[v,0]
                #xcalibraw_fe,ycalibraw_fe=hist(0,rangescal,stepcal,asiccal_fe,'stretcher')
                #xlines_fe=([5.89,6.49])
                #with sns.plotting_context("talk"):
                #    fig4=plt.figure(figsize=(16,6))
                #    plt.suptitle('QUADRANT '+ASIC)
        #
                #    specPlot(xcalibraw_fe,ycalibraw_fe,rangescal,v,2,xlines_fe,windowcal,'stretcher')
    #
                #    plt.savefig('./Quad'+ASIC+'/'+ASIC+'_XSPEC_CH_'+str(v)+'_Fe55_raw.png')
                #    plt.close(fig4)
    

        filecal.close()

        specx=ev_calib[ev_calib >= 2.0]


        table_hdu = fits.BinTableHDU.from_columns([fits.Column(name='X_evt',format='D',array=specx)])
        table_hdu.writeto('./Quad'+ASIC+'/Spectrum_X_'+ASIC+'.fits',overwrite=True)


    with sns.plotting_context("paper"):

        fig2=plt.figure(figsize=(16,6))
        plt.suptitle('QUADRANT '+ASIC)
        
        gmean,omean=gainPlot(ASIC,N_channel,'stretcher',gainv,offsetv,fwhm,CHst,Off_CHst)

        plt.savefig('./Quad'+ASIC+'/'+ASIC+'_GAIN_OFFSET.png')
        plt.close(fig2)
    
