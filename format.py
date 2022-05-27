from astropy.io import fits
import numpy as np
from config import OUTDIR

INFILE = r"D:\Dropbox\Progetti\Lab_stuff_IAPS\20220520_BEE_CC\data\20220520_161628_hermes_event_data_QABCD_55Fe109Cd137Cs_LV0d5.fits"
OUTFILE = OUTDIR + "\\bigtable.fits"

with fits.open(INFILE, memmap=True, mode='denywrite') as hdulist:
    intable = hdulist[1].data

nmult = intable.field("nmult")
N = np.size(nmult)

QUADID = intable.field("QUADID")

TRG = np.zeros((N, 6), dtype=int)
ADC = np.zeros((N, 6), dtype=int)

for i in range(6):
    TRG[:, i] = intable.field("CHANNEL{:d}".format(i))
    ADC[:, i] = intable.field("ADC{:d}".format(i))

big_matrix = np.zeros((N, 32), dtype=int)

mask_quadA = np.where(QUADID == 0)[0]
mask_quadB = np.where(QUADID == 1)[0]
mask_quadC = np.where(QUADID == 2)[0]
mask_quadD = np.where(QUADID == 3)[0]

for k in range(6):
    for i in range(N):
        big_matrix[i, TRG[i, k]] = ADC[i, k]

prim = fits.PrimaryHDU()

columns_a = []
columns_b = []
columns_c = []
columns_d = []

for i in range(32):
    columns_a.append(fits.Column(name="CH_{:02d}".format(i), array=big_matrix[mask_quadA, i], format='I'))
    columns_b.append(fits.Column(name="CH_{:02d}".format(i), array=big_matrix[mask_quadB, i], format='I'))
    columns_c.append(fits.Column(name="CH_{:02d}".format(i), array=big_matrix[mask_quadC, i], format='I'))
    columns_d.append(fits.Column(name="CH_{:02d}".format(i), array=big_matrix[mask_quadD, i], format='I'))

ta = fits.BinTableHDU.from_columns(columns_a)
tb = fits.BinTableHDU.from_columns(columns_b)
tc = fits.BinTableHDU.from_columns(columns_c)
td = fits.BinTableHDU.from_columns(columns_d)

infitsf = fits.HDUList([prim, ta, tb, tc, td])
infitsf.writeto(OUTFILE, overwrite=True)

infitsf.close()
