#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Campana, R.; Ceraudo, F.; Della Casa, G.; Dilillo, G.; Marchesini, E. J.
"""

import numpy as np
import re
from lmfit.models import GaussianModel, LinearModel, PolynomialModel, Model
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import statistics


def specPlot(x,y,ranges,ch,npeak,lines,window,mode):

    if mode=='shaper' or mode=='Shaper':
        labels='Shaper'
        colormain='blue'
        colorlines='yellow'
    if mode=='stretcher' or mode=='Stretcher':
        labels='Stretcher'
        colormain='red'
        colorlines='green'



    plt.title('CALIBRATION CH_'+str(ch))
    plt.xlim(ranges[0],ranges[1])
    plt.xlabel('Energy [keV]')
    plt.ylabel('N')
    plt.plot(x,y,color=colormain,linewidth=0.8,label=labels)

    for i in range(0,npeak):

        plt.axvspan(lines[i]-window,lines[i]+window,linestyle="dashed",color=colorlines,label=str(lines[i]),linewidth=0.6,alpha=0.5)
    #plt.axvspan(59.0,60.0,linestyle="dashed",color="purple",label="59.5 keV",linewidth=0.6,alpha=0.5)
    #plt.axvspan(57.70,57.72,linestyle="dashed",color="brown",label="59.5 keV -3%",linewidth=0.6,alpha=0.5)

    plt.legend(loc="upper right",fontsize="x-small",frameon=False)



def gainPlot(asic,ch,mode,gain,offset,fwhm,onchannels,offchannels):

    if mode=='shaper' or mode=='Shaper':
        labels='Shaper'
        mark='s'
        colormain='blue'
        colorsec='royalblue'
        colorthird='mediumturquoise'
    
    elif mode=='stretcher' or mode=='Stretcher':
        labels='Stretcher'
        mark='o'
        colormain='red'
        colorsec='indianred'
        colorthird='lightpink'
    
    else:
        labels=''
        mark='x'
        colormain='green'
        colorsec='mediumseagreen'
        colorthird='palegreen'



    for i in (offchannels):
        gain[i,:]=None 
        offset[i,:]=None 
        fwhm[i,:,:]=None 


    K,N,L = np.shape(fwhm)


    fmean=np.empty([N])
    fmedian=np.empty([N])
    gmean=np.empty([2])
    omean=np.empty([2])

    for i in range(0,N):
        fmean[i]=np.nanmean(fwhm[:,i,0])

    gmean[0]=statistics.mean(gain[onchannels,0])
    gmean[1]=statistics.mean(gain[onchannels,1])
    omean[0]=statistics.mean(offset[onchannels,0])    
    omean[1]=statistics.mean(offset[onchannels,1])

    plt.subplots_adjust(wspace=0.3,hspace=0.5)

    if N==2 or N==3:
        plt.subplot(2,N,1)
    elif N==4 or N==5:
        plt.subplot(2,N,2)
    else:
        plt.subplot(2,N,3)

    plt.xlim(-0.5,31)

    plt.title('GAIN vs. CH')
    
    plt.errorbar(ch, gain[:,0], yerr=gain[:,1], color=colormain, fmt=mark, label=labels,alpha=0.3)
    plt.xticks(np.arange(0, 32, step=4)) 
    
    plt.fill_between(np.arange(-2,34),gmean[0]-(gmean[0]/5.),gmean[0]+(gmean[0]/5.),color=colorthird,alpha=0.5,label='Mean '+labels+' +/- 20%')
    handles,labels=plt.gca().get_legend_handles_labels()
    plt.legend(labels,loc='upper right',framealpha=1,fontsize='x-small')
    
    plt.xlabel('BE CH')
    plt.ylabel('GAIN')
    
    
    if N==2 or N==3:
        plt.subplot(2,N,2)
    elif N==4 or N==5:
        plt.subplot(2,N,4)
    else:
        plt.subplot(2,N,6)
    
    plt.xlim(-0.5,31)
    plt.title('OFFSET vs. CH')
    plt.ylabel('OFFSET [instrum. units]')

    plt.errorbar(ch, offset[:,0], yerr=offset[:,1], color=colormain, fmt=mark, alpha=0.3)
    plt.xticks(np.arange(0, 32, step=3))  # Set label locations.
    
    
    plt.axhline(omean[0],color=colorsec)
    

    plt.tick_params(axis='y', which='both', right=True,left=False,bottom=False,top=False,labelleft=False, labelright=True)
    plt.xlabel('BE CH')

    for i in range(0,N):
        plt.subplot(2,N,N+1+i)
        
        plt.xlim(-0.5,31)

        plt.ylim(np.nanmin(fwhm[:,i,0])-3*(np.nanmax(abs(fwhm[:,i,1]))),np.nanmax(fwhm[:,i,0])+3*(np.nanmax(abs(fwhm[:,i,1]))))

        #plt.ylim(min(np.logical_not(np.isnan(fwhm[:,i])))-window,max(np.logical_not(np.isnan(fwhm[:,i])))+window)
        plt.ylabel('FWHM [instrum. units]')

        plt.title('Fit #'+str(i+1))
        
        plt.errorbar(ch, fwhm[0:32,i,0], yerr=fwhm[0:32,i,1],color=colormain, fmt=mark,alpha=0.3)
        plt.xticks(np.arange(0, 32, step=3))  # Set label locations.
        
        plt.axhline(fmean[i],color=colorsec)
        plt.tick_params(axis='y', which='both', right=True,left=False,bottom=False,top=False,labelleft=False, labelright=True)
        plt.xlabel('BE CH')

    return gmean,omean
    


def linPlot(mode,ranges,ch,xadc,xlines,gain,offset,calib_units, **kwargs):
    """
    Produces a plot with all the relevant information (centroid, fitting function, calibration function)
    """
    if mode=='shaper' or mode=='Shaper':
        labels='Shaper'
        mark='s'
        colormain='blue'
        colorsec='royalblue'
        colorthird='mediumturquoise'
    
    elif mode=='stretcher' or mode=='Stretcher':
        labels='Stretcher'
        mark='o'
        colormain='red'
        colorsec='indianred'
        colorthird='lightpink'
    
    else:
        labels=''
        mark='x'
        colormain='green'
        colorsec='mediumseagreen'
        colorthird='palegreen' 
    
    fig, ax = plt.subplots(**kwargs)
    ax.set_xlim(ranges[0],ranges[1])
    ax.set_title('LINEARITY CH '+str(ch))
    ax.set_ylabel('Amplitude ['+calib_units+']')
    ax.set_xlabel('Centroid [instrum. units]')
    
    props = dict(boxstyle='round', facecolor='silver', alpha=0.5)
    txt_string = "Gain: {:.3f}\nOffset: {:.1f}".format(gain, offset)
    ax.text(0.05, 0.95, txt_string, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    ax.errorbar(xadc[:,0], xlines, xerr=xadc[:,1], fmt=mark, color=colorsec,markersize=8,alpha=0.5,label=labels)
    ax.plot(float(gain)*np.array(xlines)+float(offset), xlines, lw=2, color=colormain,label='Best fit')
    ax.legend(loc="upper right",fontsize="x-small",frameon=False)
    #plt.show()
    return fig, ax


def fano(E):
    """
    Returns Fano noise in eV FWHM
    Input: energy in keV
    """
    return np.sqrt(E*1000./3.6*0.118)*2.35*3.6


def lineFitter(x, y, limits, bkg=None, verbose=True):  
    """
    Fits a Gaussian line between start and stop
    Input:  x, y: input arrays
            limits = (start,stop): tuple with fit boundaries
            bkg (optional): polynomial background. If not None, should be polynomial degree (up to 7th deg.)
    Output: lmfit result object
    """
    start = limits[0]
    stop = limits[1]
    y_err = np.sqrt(y)
    x_start = np.where(x > start)[0][0]
    x_stop  = np.where(x < stop)[0][-1]
    x_fit = x[x_start:x_stop]
    y_fit = y[x_start:x_stop]
    y_fit_err = y_err[x_start:x_stop]

    if bkg is not None:
        mod = GaussianModel() + PolynomialModel(bkg, prefix='bkg_')
    else:
        mod = GaussianModel()
    pars = mod.guess(y_fit , x=x_fit)
    center = x_fit[np.argmax(y_fit)]
    pars['center'].set(center, min=start, max=stop)
    result = mod.fit(y_fit, pars, x=x_fit, weights=y_fit_err)
   

    x_fine = np.linspace(x[0],x[-1],len(x)*100)
    fitting_curve = mod.eval(x=x_fine, \
                         amplitude    =  result.best_values['amplitude'], \
                         center       =  result.best_values['center'], \
                         sigma        =  result.best_values['sigma'] )
        
    return result,start,stop,x_fine,fitting_curve


def dataprep(outfile_path, datafile, fitsfile, ASIC):
    """
    Read the file with the expected lines and energies to be fitted the spectrum and prepare both input and output files.
    Also it creates all the arrays that will be used for the analysis.

    Input: datafile : .txt file of two columns with the name of the line and the peak energy
           fitsfile : .fits file with the data for the spectrum
           ASIC : quadrant in use (A, B, C, or D)
       
    Output: outputfile : .txt file to be filled for every ASIC, the header is created in the function
            counts_data : data collected for the spectra from the FITS file
            line_data = {line_name, line_details} : dictionary, respectively the name and an array with (energy, color) for every peak  
    """
    color = {0: "orange", 1: "cornflowerblue", 2: "indianred", 3: "green", 4: "yellow", 5:"purple", 6:"brown", 7:"black"}
    # TODO: what happens if you have more lines than colors?
    
    line_data = {}
    
    infile = open(datafile)
    for i,line in enumerate(infile):
        lname, lenergy = line.split()
        line_data[lname] = (float(lenergy), color[i])

    hdulist = fits.open(fitsfile)
    counts_data = hdulist[1].data
    hdulist.close()
    
    outputfile = open(outfile_path,'w+')
    string = "#ASIC  CH  "
    for key in sorted(line_data):
        string = string + "x_adc({0:s})  FWHM({0:s})   x_adc_err({0:s})  FWHM_err({0:s})".format(key)
    outputfile.write(string)

    calib_units = ''.join(re.findall(r'[a-zA-Z]', lname))

    return outputfile, counts_data, line_data, calib_units


def detectPeaks(x, y, w_size, n_peak, window, smooth):
    """
    Applies a smoothing to the data collected and finds the peaks in the spectra.
    
    Inputs: x, y : arrays of data
            w_size : integer to define the smoothing
            n_peak : number of peaks to be found
            window : integer defining the range around the peak
            smooth : boolean, whether or not do the smoothing
    Outputs: limit : the limits around the peak
             peak : array of the peaks found
    """
    if smooth:
        y_smooth = savgol_filter(y, w_size, 3) # window size 51, polynomial order 3
        #plt.figure()
        #plt.plot(x,y_smooth,drawstyle='steps-mid',label='smooth data')
        #plt.title("CH {:02d}".format(v))
        #plt.show()
        peak, _ = find_peaks(y_smooth, height=1, width = 1, distance=20,prominence=25)
        ytmp = y_smooth.copy()
    else:
        peak, _ = find_peaks(y,height=4, width = 5, distance=20,prominence=1)
        ytmp = y.copy()
        
    index = np.zeros(n_peak)
    
         
    # TODO: generalizzare la ricerca dei picchi a N

    for i in range(n_peak):
        index[i] = np.argmax(ytmp[peak])
        ytmp[peak[int(index[i])]] = 0
    
    indexlist=sorted([int(x) for x in index])
    
    limit = np.zeros((n_peak,2))

    for i in range(n_peak):
        limit[i] = (x[peak[indexlist[i]]]-window,x[peak[indexlist[i]]]+window)
    
    return limit, peak, indexlist
    


def clean_gamma(counts_data,  couples, threshold,keep):
    """
    Removes the data due to gammas, i.e. that are seen by both the SDD cells optically coupled with the same crystal, in order to do a proper calibration.

    Input: counts_data : array containing the value measured by every channel for every event.
           couples : array with the coupled channels.
           threshold : integer defining when a channel has revealed a photon.
    
    Output: clean_data : same array of counts_data, without the gamma events.
    """
    print('Cleaning gamma-ray events with threshold', threshold)
    print('Counts data original length', len(counts_data))


    mask = np.ones(counts_data.shape,dtype=bool) #np.ones_like(a,dtype=bool)

    original_counts=len(counts_data)

    for couple in couples:
        print("Looking for events in pair:", couple,' with threshold ', threshold)
        pair_a, pair_b = couple

        idx = np.where(np.logical_and( counts_data.field("CH_{:02d}".format(pair_a)) >= threshold, counts_data.field("CH_{:02d}".format(pair_b)) >= threshold))
        mask[idx]=False

        if keep:

            clear_data=np.empty_like(counts_data[~mask])
            clear_data=counts_data[~mask]

        else:
            clear_data=np.empty_like(counts_data[mask])
            clear_data=counts_data[mask]

    print('Counts data new length', len(clear_data))
    print('Reduced of {:.2f}%'.format( ( len(clear_data) / original_counts )*100. ))


    return clear_data
    

def guess_pedestal(bins, counts,**kwargs):
    """
    a scipy.signal.find_peaks wrapper, 
    it checks for peaks and returns the average
    of the first two it finds (if more than two exist)
    else returns None. by default find_peaks is operated
    with parameters height = 10 and distance = 20. 
    you can override these parameters or provide different ones 
    e.g., guess_pedestal(bins,counts, height = 20, width = 20) will
    call find_peaks(counts, height = 20, distance = 20, width = 20)
    
    params:
    :param bins: sequence, histogram bins
    :param counts: sequence, histogram counts
    :param kwargs: other keyword arguments are passed to find_peaks.
    :return: a number or None, pedestal value
    """
    default = {'height': 20, 'distance': 20}
    params = {k:(kwargs[k] if k in kwargs else default[k]) 
                for k in set(default)|set(kwargs)}

    peaks, *_ = find_peaks(counts, **params)
 
    if len(peaks) > 1:
        return sum(bins[peaks[:2]])/2


    return None


def hist(pedestal, ranges, step, data, automatic, mode):
    """
    Prepare the data, binning it and removing the pedestal. 

    Input: pedestal :  array given through a .txt file
           ranges, step : array and int given in the main file 
           data : array with the data to be prepared
           mode : string, shaper or stretcher
    
    Output: x, y : array of data binned and with pedestal removed
    """
    binning = np.arange(ranges[0], ranges[1] + step, step)
    x = binning[:-1] + step/2.
    y, bins = np.histogram(data, bins=binning)

    pedestal_auto=guess_pedestal(x, y)

    print('The automatic minimum value below which everything is discarded is ',pedestal_auto)
    print(' while the user-defined minimum value is', pedestal)


    if automatic:
        pedestal=pedestal_auto

    if mode == "stretcher":
        pedestal_limit = np.where(x >= pedestal)[0][0]
        y[0:pedestal_limit] = 0
            
    return x, y


def fitPeaks(x, y, limits):
    """
    Fit the peaks in the given spectrum, each in the given limits
    
    Input: 
            x, y : arrays with input spectrum
            limits : array of limits for each peak 
    Output:
            fit_results : array, rows is each peak, column is mu, mu_err, fwhm, fwhm_err
            x_fine: array of adu. useful for visualization
            fitting_curve: array of fit values. useful for visualization
    """
    n_peaks = len(limits)
    x_adc = np.zeros(n_peaks)
    x_adc_err = np.zeros(n_peaks)
    sigma = np.zeros(n_peaks)
    sigma_err = np.zeros(n_peaks)
    x_fines = []
    fitting_curves = []

    for i in range(n_peaks):
        result, start, stop, x_fine, fitting_curve = lineFitter(x, y, (limits[i][0], limits[i][1]), verbose=True)
        x_adc[i], x_adc_err[i], sigma[i], sigma_err[i] =  result.params['center'].value, result.params['center'].stderr, \
                                                    result.params['sigma'].value, result.params['sigma'].stderr
        x_fines.append(x_fine)
        fitting_curves.append(fitting_curve)

        if sigma_err[i] is None:
            sigma_err[i]=0.
    FWHM = 2.355*sigma
    FWHM_err = 2.355*sigma_err
    

    # Sanification  
    x_adc = [item if item else 0. for item in x_adc]
    x_adc_err = [item if item else 0. for item in x_adc_err]
    x_adc = [0 if np.isnan(item) else item for item in x_adc]
    x_adc_err = [0 if np.isnan(item) else item for item in x_adc_err]
    
    fit_results = np.column_stack((x_adc, x_adc_err, FWHM, FWHM_err))
    return fit_results, x_fines, fitting_curves


def calibrate(ASIC,v,line_data, fit_results, calib_units,mode='stretcher',verbose=True):
    """
    This function establish gain and offset values for each channel, with respective errors from the peaks fitted.
    
    Input:
          line_data : array containing the energy line and the color associated
          fit_results : array with the fit of the peaks
    Output:
          chi, gain, gain_err, offset, offset_err : floats
        
    """
    if mode.lower() == 'stretcher':
        colorlin='red'
        colormark='tomato'
        fmtmark='o'
    elif mode.lower() == 'shaper':
        colorlin='blue'
        colormark='cornflowerblue'
        fmtmark='s'
        
    adc = fit_results[:,0]
    adc_err = fit_results[:,1]
    energy_line = []
    for key in sorted(line_data.keys()):
        energy_line.append(line_data[key][0])
    
    energy_line.sort()

    lmod = LinearModel()
    pars = lmod.guess(adc, x=energy_line)

    try:
        resultlin = lmod.fit(adc, pars, x=energy_line, weights=adc_err)
        figcal=plt.figure(figsize=(8,6))
        resultlin.plot(datafmt='o',fitfmt='-', initfmt='--', xlabel=calib_units, ylabel='Instrum. units', yerr=None, numpoints=None, fig=figcal, data_kws=None, fit_kws=None, init_kws=None, ax_res_kws=None, ax_fit_kws=None, fig_kws=None, show_init=False, parse_complex='abs')
        plt.savefig('./Quad'+ASIC+'/'+ASIC+'_FITLIN_CH_'+str(v)+'.png')
        plt.close(figcal)

    except TypeError:
        print('xadc',x_adc)
        print('xline',x_line)
        print('xadcerr',x_adc_err)

    chi= resultlin.redchi
    gain = resultlin.params['slope'].value
    offset = resultlin.params['intercept'].value
    gain_err = resultlin.params['slope'].stderr
    offset_err =resultlin.params['intercept'].stderr
    
    return chi, gain, gain_err, offset, offset_err, energy_line
