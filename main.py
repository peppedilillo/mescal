from source import upaths
import numpy as np
import matplotlib.pyplot as plt

from configs import fm1
from source.structs import add_evtype_flag_to, get_from
from source.specutilities import detect_peaks, fit_peaks, calibrate


asics = 'D'
lines = [5.9, 22.1, 24.9]
start, stop, step = 15000, 22000, 10
nbins = int((stop - start) / step)
fitsfile = upaths.DATADIR.joinpath('20220525_125447_hermes_event_data_QABCD_55Fe_109Cd_137Cs_LV0d5.fits')

on_channels = (lambda data, q: np.unique(data[data['QUADID'] == q]['CHN']))

if __name__ == '__main__':
    data = get_from(fitsfile)
    for asic in asics:
        couples = fm1.get_couples(asic)
        quad_df = add_evtype_flag_to(data[data['QUADID'] == asic], couples)

        for ch in on_channels(data, asic):
            ch_data = quad_df[ (quad_df['CHN'] == ch)]
            counts, bins = np.histogram(ch_data['ADC'], range=(start, start + nbins * step), bins=nbins)
            limits = detect_peaks(bins, counts, lines)
            centers, center_errs, fwhms, fwhm_errs, *amps = fit_peaks(bins, counts, limits)
            gain, gain_err, offset, offset_err, chi2 = calibrate(centers, center_errs, lines)

            fig, ax = plt.subplots(1, 1)
            ax.step(bins[:-1], counts)
            ax.set_title(ch)
            for c,l in zip(centers, limits):
                ax.axvline(c, color='red', linestyle='dotted')
                ax.axvspan(*l, color = 'r', alpha = 0.1)
            fig.show()
