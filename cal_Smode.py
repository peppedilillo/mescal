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

# visualize=True to visualize every intermediate step through several plots
visualize = False

# smooth=True to apply an smoothing filter to the raw data in order to easily find the peaks
# Note that every other operation, including calibration, is always done with the *non* smoothed data
smooth = True



# ranges[1,2] defines the lower and upper limits of the spectra, in ADC channels
# while step defines how many channels will be contained inside one single bin
ranges = [1000,2400]
step = 2

ranges_sum_el=[11000, 19000]
ranges_el=[6200, 14000]

hist_el=[3000,19000]
step_el=100

asics = ['C']
scintfitsfiles = ['QC_Cs137.fits']


for i in range(len(asics)):
    ASIC = asics[i]
    scintfitsfile = scintfitsfiles[i]
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
    
    # Reading the data from the scintillator file
    hdulist = fits.open(scintfitsfile)
    counts_data = hdulist[1].data
    hdulist.close()


    with open("Pedestal_%s.txt" % ASIC, 'r') as f:
        pedestal = np.loadtxt(islice(f,1), delimiter=" ", dtype= float)
        couples = np.loadtxt(f, delimiter=" ", dtype= int)

    events_electrons = np.empty((32, len(counts_data)))
    light_output = np.zeros(32)
    light_output_err = np.zeros(32)

    for channel in sorted(calibration.keys()):

        print("Analyzing ASIC {:s} channel {:02d}\n".format(ASIC, channel))
        channel_data = counts_data.field("CH_{:02d}".format(channel))
        # building the raw spectrum
        x, y = hist(pedestal[channel], ranges, step, channel_data, 'stretcher')
        
        plt.figure()
        plt.plot(x,y,drawstyle='steps-mid')
        plt.title("CH {:02d}".format(channel))
        # Fitting peaks for the given spectrum
        fit_results = fitPeaks(x, y, [[pedestal[channel], ranges[1]]], visualize=visualize)
        #fit_results = fitPeaks(x, y, [[pedestal[channel], 1800]], visualize=visualize)

        #print(fit_results)
        center = fit_results[0][0]
        center_err = fit_results[0][1]

        events_electrons[channel]= (channel_data-calibration[channel][1])/calibration[channel][0] *1000/3.65 
        
        plt.axvline(center, c='g')     
        if visualize:
            plt.show()


        plt.close()

        light_output[channel] = (center-calibration[channel][1])/calibration[channel][0] *1000/3.65 /661.657
        light_output_err[channel] = np.fabs( (center_err)/calibration[channel][0] ) *1000/3.65 /661.657


        print("Light output: {:.1f} +- {:.2f}".format(light_output[channel], light_output_err[channel]))
        output.write("{:02d}\t{:.1f}\t{:.2f}\n".format(channel,light_output[channel], light_output_err[channel]))



    output.close()

    i=0
    for couple in couples:

        ev_el_a=events_electrons[couple[0]]
        ev_el_b=events_electrons[couple[1]]

        sum_electrons= [ ev_el_a[i]+ev_el_b[i] for i in range (min(len(ev_el_a),len(ev_el_b))) if ev_el_a[i]>1 and ev_el_b[i]>1] 

    
        xg,yg=hist(0, hist_el,step_el,  sum_electrons, 'stretcher')
        xa,ya=hist(0, hist_el,step_el,  ev_el_b, 'stretcher')
        xb,yb=hist(0, hist_el,step_el,  ev_el_a, 'stretcher')
        
        fit_resultsg = fitPeaks(xg, yg, [ranges_sum_el], visualize=visualize)
        fit_resultsa = fitPeaks(xa, ya, [ranges_el], visualize=visualize)
        fit_resultsb = fitPeaks(xb, yb, [ranges_el], visualize=visualize)




        table_hdu = fits.BinTableHDU.from_columns([fits.Column(name='G_evt_el_CH_'+str(couple[0])+'_CH_'+str(couple[1]),format='D',array=sum_electrons)])
        table_hdu.writeto('./Quad'+ASIC+'/Spectrum_G_'+ASIC+'CH_'+str(couple[0])+'_CH_'+str(couple[1])+'.fits',overwrite=True)
         

        spectrum = np.empty((15, len(sum_electrons)))

        if sum(sum_electrons) != 0 :

            table_hdu = fits.BinTableHDU.from_columns([fits.Column(name='G_evt_en_CH_'+str(couple[0])+'_'+str(couple[1]),format='D',array=spectrum[i])])
            table_hdu.writeto('./Quad'+ASIC+'/Spectrum_G_'+ASIC+'_CH_'+str(couple[0])+'_CH_'+str(couple[1])+'.fits',overwrite=True)
            

            fig4=plt.figure(figsize=(16,6))
            plt.title('EVENTS [ELECTRONS] FOR QUADRANT '+ASIC+' [CH_'+str(couple[0])+' & CH_'+str(couple[1])+']')

            plt.xlabel('Events [electrons]')            
            plt.plot(xg,yg,c='black',label='Sum')
            plt.axvline((fit_resultsg[0][0]),c='black')
            plt.plot(xa,ya,c='r',label=str(couple[0]))
            plt.axvline((fit_resultsa[0][0]),c='r')
            plt.plot(xb,yb,c='b',label=str(couple[1]))
            plt.axvline((fit_resultsb[0][0]),c='b') 
            plt.legend()   
            if visualize:
                plt.show()
            plt.savefig('./Quad'+ASIC+'/Spec_el_CH_'+str(couple[0])+'_CH_'+str(couple[1])+'.png')

            plt.close(fig4)



            spectrum[i] = sum_electrons/(light_output[couple[0]]+light_output[couple[1]])  
            xg_en, yg_en = hist(0, [20,800], step, spectrum[i], 'stretcher')



            fig5=plt.figure(figsize=(16,6))
            plt.title('EVENTS [ENERGY] FOR QUADRANT '+ASIC+' [CH_'+str(couple[0])+' & CH_'+str(couple[1])+']')
            plt.axvline(661.657, c='g',label='661.657 keV')
            plt.xlabel('Energy [keV]')
            plt.plot(xg_en,yg_en,c='black')
            
            if visualize:
                plt.show()
            plt.savefig('./Quad'+ASIC+'/Spec_en_CH_'+str(couple[0])+'_CH_'+str(couple[1])+'.png')

            plt.close(fig5)


            i += 1


    final_spectrum=spectrum[spectrum >= 20.0]


    table_hdu = fits.BinTableHDU.from_columns([fits.Column(name='G_evt',format='D',array=final_spectrum)])
    table_hdu.writeto('./Quad'+ASIC+'/Spectrum_G_'+ASIC+'.fits',overwrite=True)



    x_cal, y_cal = hist(0, [20,800], step, final_spectrum, 'stretcher')
    
    fig6=plt.figure(figsize=(16,6))
    plt.title('S-mode Spectrum')
    plt.axvline(661.657, c='g', label='661.657 keV')
    plt.plot(x_cal[1:],y_cal[1:],drawstyle='steps-mid')
    plt.xlabel('Energy [keV]')

    if visualize:
        plt.show()
    plt.savefig('./Quad'+ASIC+'/Spectrum_G.png')

    plt.close(fig6)
