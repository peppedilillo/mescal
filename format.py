from astropy.io import fits
import dask.array as da
import numpy as np

fitsfile='20220523_173237_hermes_event_data_QABCD_15deg_thr95_55Fe_109Cd_137Cs_LV0d5.fits'
#fitsfile='Prova_short.fits'


hdulist = fits.open(fitsfile,memmap=True,mode='denywrite')
intable = hdulist[1].data
hdulist.close()

NMULT = intable.field("NMULT")
N=np.size(NMULT)

QUADID= intable.field("QUADID")

TRG=np.zeros((N,6), dtype=int)
ADC=np.zeros((N,6), dtype=int)

for i in range(6):
   TRG[:,i]=intable.field("CHANNEL{:d}".format(i))
   ADC[:,i]=intable.field("ADC{:d}".format(i))



big_matrix = np.zeros((N,32), dtype=int)

mask_quadA = np.where(QUADID==0)[0]
mask_quadB = np.where(QUADID==1)[0]
mask_quadC = np.where(QUADID==2)[0]
mask_quadD = np.where(QUADID==3)[0]

print(len(mask_quadA))
print(len(mask_quadB))
print(len(mask_quadC))
print(len(mask_quadD))



for k in range(6):
    print(k)
    for i in range(N):
        big_matrix[i,TRG[i,k]] = ADC[i,k]



prim = fits.PrimaryHDU()


columns_a = []
columns_b = []
columns_c = []
columns_d = []

for i in range(32):
    columns_a.append(fits.Column(name="CH_{:02d}".format(i), array=big_matrix[mask_quadA,i], format='I'))
    columns_b.append(fits.Column(name="CH_{:02d}".format(i), array=big_matrix[mask_quadB,i], format='I'))
    columns_c.append(fits.Column(name="CH_{:02d}".format(i), array=big_matrix[mask_quadC,i], format='I'))
    columns_d.append(fits.Column(name="CH_{:02d}".format(i), array=big_matrix[mask_quadD,i], format='I'))



ta = fits.BinTableHDU.from_columns(columns_a)
tb = fits.BinTableHDU.from_columns(columns_b)
tc = fits.BinTableHDU.from_columns(columns_c)
td = fits.BinTableHDU.from_columns(columns_d)

infitsf= fits.HDUList([prim,ta,tb,tc,td])
infitsf.writeto('test2.fits',overwrite=True)

infitsf.close()
