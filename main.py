import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs import fm1
from source.structs import add_evtype_flag_to, get_from
from source.specutilities import detect_peaks, fit_peaks, calibrate
from source import upaths

asics = 'D'
lines = [5.9, 22.1, 24.9]
start, stop, step = 15000, 24000, 10
nbins = int((stop - start) / step)
fitsfile = upaths.DATADIR.joinpath('20220525_125447_hermes_event_data_QABCD_55Fe_109Cd_137Cs_LV0d5.fits')

on_channels = (lambda data, q: np.unique(data[data['QUADID'] == q]['CHN']))

lines = {'Fe 5.9 keV': 5.9, 'Cd 22.1 keV': 22.1, 'Cd 24.9 keV': 24.9}
fit_params = ["center", "center_err", "fwhm", "fwhm_err", "amp", "amp_err"]
cal_params = ["gain", "gain_err", "offset", "offset_err", "chi2"]

# fig, ax = plt.subplots(1, 1)
# ax.step(bins[:-1], counts)
# ax.set_title(ch)
# for c, l in zip(centers, limits):
#     ax.axvline(c, color='red', linestyle='dotted')
#     ax.axvspan(*l, color='r', alpha=0.1)
# fig.show()

if __name__ == '__main__':
    results_fit, results_cal = {}, {}

    lines_keys, lines_values = zip(*lines.items())
    data = get_from(fitsfile)
    for asic in asics:
        results_fit_asic, results_cal_asic = {}, {}
        couples = fm1.get_couples(asic)
        quad_df = add_evtype_flag_to(data[data['QUADID'] == asic], couples)

        for ch in on_channels(data, asic):
            ch_data = quad_df[(quad_df['CHN'] == ch) & (quad_df['EVTYPE'] == 'X')]
            counts, bins = np.histogram(ch_data['ADC'], range=(start, start + nbins * step), bins=nbins)
            limits = detect_peaks(bins, counts, lines_values)
            centers, center_errs, *etc = fit_peaks(bins, counts, limits)
            gain, gain_err, offset, offset_err, chi2 = calibrate(centers, center_errs, lines_values)

            results_fit_asic[ch] = np.concatenate((centers, center_errs, *etc))
            results_cal_asic[ch] = np.array((gain, gain_err, offset, offset_err, chi2))
        results_fit[asic] = pd.DataFrame(results_fit_asic, index=pd.MultiIndex.from_product((fit_params, lines_keys))).T
        results_cal[asic] = pd.DataFrame(results_cal_asic, index=cal_params).T
