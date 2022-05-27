#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Campana, R.; Ceraudo, F.; Della Casa, G.; Dilillo, G.; Marchesini, E. J.
"""

import numpy as np
import lmfit
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
To run this script, a file named calparams_X_*.txt, * being A, B, C or D (quadrant) should be provided
which should include:
1. a line with the threshold below which all data are discarded (pedestal) for each channel, including those turned off
2. N lines with the coupled channels, i.e.:
1 2
3 4

if channels 01 and 02 are coupled, and 03 and 04 are coupled."""


# visualize=True to visualize every intermediate step through several plots
visualize = True

# smooth=True to apply an smoothing filter to the raw data in order to easily find the peaks
# Note that every other operation, including calibration, is always done with the *non* smoothed data
# this is not very important in cal_Smode
smooth = True

# threshold defines the threshold above which events are going to be read by the "clean gamma" function
# (meaning: x is considered a gamma event if channel A > 10 AND channel B > 10, A,B scintillator couple)
threshold=1

# keep commands if clean gamma is going to *keep* the gamma rays (and thus discard the X-rays) (True)
# or if clean gamma is going to *discard* the gamma rays (and thus keep the X-rays) (False)
# generally speaking, cal_Smode operates with keep=True
keep=True

# note that CLEAN GAMMA IS ALWAYS PERFORMED IN cal_Smode
# it is only optional in cal_Xmode!


automatic=False

# ranges[1,2] defines the lower and upper limits of the spectra, in ADC channels
# while step defines how many channels will be contained inside one single bin
ranges = [16000,22000]
step = 7

asics = ['A','B','C','D']

fitsfiles = '20220525_16.fits'



# this section is needed if visualizing intermmediate, electron spectra (go to lines 145 and 161)
#ranges_sum_el=[3800000, 8700000]
#ranges_el=[6200, 14000]
##
#hist_el=[3000,19000]
#step_el=100

hdulist = fits.open(fitsfiles)

for i in range(len(asics)):
    ASIC = asics[i]
    calibfile = './Quad'+ASIC+'/Calibration_Stretcher_ASIC_{:s}.txt'.format(ASIC)
    
    outputfile = './Quad'+ASIC+'/LightOutput_ASIC_{:s}.txt'.format(ASIC)
    output = open(outputfile, "w")
    output.write("# CH\tLight_output\tLight_output_err\n")

    # Reading the data from the calibration file
    calibration = {}
    f = open(calibfile, 'r')
    for line in f.readlines()[1:]:
        channel, gain, gain_err, offset, offset_err, chi2 = line.split()
        calibration[int(channel)] = (float(gain), float(offset))
    f.close()
    
    # Reading the data from the scintillator file from the corresponding ASIC
    counts_data = hdulist[i+1].data


    with open("calparams_S_%s.txt" % ASIC, 'r') as f:
        pedestal = np.loadtxt(islice(f,1), delimiter=" ", dtype= float)
        couples = np.loadtxt(f, delimiter=" ", dtype= int)


    counts_data=clean_gamma(counts_data, couples, threshold,keep)



    events_electrons = np.empty((32, len(counts_data)))
    light_output = np.zeros(32)
    light_output_err = np.zeros(32)

    for channel in sorted(calibration.keys()):

        print("Analyzing ASIC {:s} channel {:02d}\n".format(ASIC, channel))
        channel_data = counts_data.field("CH_{:02d}".format(channel))
        

        # building the raw spectrum
        #print()
        x, y = hist(pedestal[channel], ranges, step, channel_data, automatic,'stretcher')
        
        plt.figure()
        plt.plot(x,y,drawstyle='steps-mid')
        plt.title("CH {:02d}".format(channel))
        

        # Fitting peaks for the given spectrum
        fit_results = fitPeaks(x, y, [[pedestal[channel], ranges[1]]], visualize=visualize)

        center = fit_results[0][0]
        center_err = fit_results[0][1]

        events_electrons[channel]= (channel_data-calibration[channel][1])/calibration[channel][0] *1000/3.65 
        
        plt.axvline(center, c='g')     
        if visualize:
            plt.show()


        plt.close()


        #obtaining the resulting light output
        light_output[channel] = (center-calibration[channel][1])/calibration[channel][0] *1000/3.65 /661.657
        light_output_err[channel] = np.fabs( (center_err)/calibration[channel][0] ) *1000/3.65 /661.657


        print("Light output: {:.1f} +- {:.2f}".format(light_output[channel], light_output_err[channel]))
        output.write("{:02d}\t{:.1f}\t{:.2f}\n".format(channel,light_output[channel], light_output_err[channel]))



    output.close()


    # creating a dictionary of lists where to save each spectrum for each couple (16 couples)
    spectrum = {}
    for i in range(16):
        spectrum["spec_ch%s" %i] = []


    automatic=False
    i=0
    for couple in couples:

        ev_el_a=events_electrons[couple[0]]
        ev_el_b=events_electrons[couple[1]]

        sum_electrons= [ ev_el_a[i]+ev_el_b[i] for i in range (min(len(ev_el_a),len(ev_el_b))) if ev_el_a[i]>1 and ev_el_b[i]>1] 

    
        # this section is to build intermediate spectra (in electrons per channel, and in electrons per summed-couple)
        #xg,yg=hist(0, hist_el,step_el,  sum_electrons, 'stretcher')
        #xa,ya=hist(0, hist_el,step_el,  ev_el_b, 'stretcher')
        #xb,yb=hist(0, hist_el,step_el,  ev_el_a, 'stretcher')
        #
        #fit_resultsg = fitPeaks(xg, yg, [ranges_sum_el], visualize=visualize)
        #fit_resultsa = fitPeaks(xa, ya, [ranges_el], visualize=visualize)
        #fit_resultsb = fitPeaks(xb, yb, [ranges_el], visualize=visualize)

        # these fits save a spectrum for each scintillator couple
        table_hdu = fits.BinTableHDU.from_columns([fits.Column(name='G_evt_el_CH_'+str(couple[0])+'_CH_'+str(couple[1]),format='D',array=sum_electrons)])
        table_hdu.writeto('./Quad'+ASIC+'/Spectrum_G_'+ASIC+'CH_'+str(couple[0])+'_CH_'+str(couple[1])+'.fits',overwrite=True)
         
        if sum(sum_electrons) != 0 :

   
            # this section is to build intermediate spectra (in electrons per channel, and in electrons per summed-couple)

            #fig4=plt.figure(figsize=(16,6))
            #plt.title('EVENTS [ELECTRONS] FOR QUADRANT '+ASIC+' [CH_'+str(couple[0])+' & CH_'+str(couple[1])+']')
            #plt.xlabel('Events [electrons]')            
            #plt.plot(xg,yg,c='black',label='Sum')
            #plt.axvline((fit_resultsg[0][0]),c='black')
            #plt.plot(xa,ya,c='r',label=str(couple[0]))
            #plt.axvline((fit_resultsa[0][0]),c='r')
            #plt.plot(xb,yb,c='b',label=str(couple[1]))
            #plt.axvline((fit_resultsb[0][0]),c='b') 
            #plt.legend()   
            #if visualize:
            #    plt.show()
            #plt.savefig('./Quad'+ASIC+'/Spec_el_CH_'+str(couple[0])+'_CH_'+str(couple[1])+'.png')
            #plt.close(fig4)



            # this section builds a spectrum in keV for each scintillator couple

            spectrum[i] = sum_electrons/(light_output[couple[0]]+light_output[couple[1]])

            xg_en, yg_en = hist(0, [20,800], step, spectrum[i], automatic,'stretcher')

            fig5=plt.figure(figsize=(16,6))
            plt.title('EVENTS [ENERGY] FOR QUADRANT '+ASIC+' [CH_'+str(couple[0])+' & CH_'+str(couple[1])+']')
            plt.axvline(661.657, c='g',label='661.657 keV')
            plt.xlabel('Energy [keV]')
            plt.plot(xg_en,yg_en,c='black')
            
            if visualize:
                plt.show()
            plt.savefig('./Quad'+ASIC+'/Spec_en_CH_'+str(couple[0])+'_CH_'+str(couple[1])+'.png')

            plt.close(fig5)


            table_hdu = fits.BinTableHDU.from_columns([fits.Column(name='G_evt_en_CH_'+str(couple[0])+'_'+str(couple[1]),format='D',array=spectrum[i])])
            table_hdu.writeto('./Quad'+ASIC+'/Spectrum_G_'+ASIC+'_CH_'+str(couple[0])+'_CH_'+str(couple[1])+'.fits',overwrite=True)
         

            i += 1


    #this section builds a single event list in keV for *all* couples together, "stacked", and filters it for energies > 20 keV
    filtered_spectra= [x for v in spectrum.values() for x in v if float(x)>=20.0]

    final_spectrum=np.array(filtered_spectra)


    table_hdu = fits.BinTableHDU.from_columns([fits.Column(name='G_evt',format='D',array=final_spectrum)])
    table_hdu.writeto('./Quad'+ASIC+'/Spectrum_G_'+ASIC+'.fits',overwrite=True)


    #final, stacked plot of all couples
    x_cal, y_cal = hist(0, [20,800], step, final_spectrum, automatic, 'stretcher')
    
    fig6=plt.figure(figsize=(16,6))
    plt.title('S-mode Spectrum')
    plt.axvline(661.657, c='g', label='661.657 keV')
    plt.plot(x_cal[1:],y_cal[1:],drawstyle='steps-mid')
    plt.xlabel('Energy [keV]')

    if visualize:
        plt.show()
    plt.savefig('./Quad'+ASIC+'/Spectrum_G.png')

    plt.close(fig6)
