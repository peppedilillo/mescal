#
#from astropy.io import fits
#
#fits_image_filename = fits.util.get_testdata_filepath('test0.fits')
#
#hdul = fits.open(fits_image_filename)
#hdul.info()
#
#
#
#hdr = hdul[1].header
#hdr.set('observer', 'Edwin Hubble')
#
#hdul.info()
#print(repr(hdr))
#hdr.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)
#
#hdr.append(('coso2', 'OMIT', 'Dark Image Subtraction'), end=True)
#print(repr(hdr))


from astropy.io import fits

import numpy as np

fitsfile='Fe55v5_short.fits'

hdulist = fits.open(fitsfile)
intable = hdulist[1].data
hdulist.close()

Nmult = intable.field("NMULT")
N=np.size(Nmult)




QuadID= intable.field("QUADID")

TRG=np.empty((N,6))
ADC=np.empty((N,6))

# FAILED ATTEMPT TO MAKE THIS SHORTER
#for i in range(6)
#
#    TRG[:,i]=intable.field("CHANNEL" i)
#    ADC[:,i]=intable.field("ADC" i)


TRG[:,0]=intable.field("CHANNEL0")
TRG[:,1]=intable.field("CHANNEL1")
TRG[:,2]=intable.field("CHANNEL2")
TRG[:,3]=intable.field("CHANNEL3")
TRG[:,4]=intable.field("CHANNEL4")
TRG[:,5]=intable.field("CHANNEL5")

ADC[:,0]=intable.field("ADC0")
ADC[:,1]=intable.field("ADC1")
ADC[:,2]=intable.field("ADC2")
ADC[:,3]=intable.field("ADC3")
ADC[:,4]=intable.field("ADC4")
ADC[:,5]=intable.field("ADC5")

col=np.ones((N, 1))
col=-1*col


# failed attempt to make the next part shorter

#cols=np.zeros((N,32))
#
#print(cols)
#
#
#
#
#for i in range(32):
#
#    cols[:,i]=fits.Column(name='culo', array=col, format='D')
#    #cols[i] = fits.Column(name="CH_{:02d}".format(i), array=col, format='D')

#t = fits.BinTableHDU.from_columns(cols[0:31])


c0=fits.Column(name="CH_00", array=col, format='D')
c1=fits.Column(name="CH_01", array=col, format='D')
c2=fits.Column(name="CH_02", array=col, format='D')
c3=fits.Column(name="CH_03", array=col, format='D')
c4=fits.Column(name="CH_04", array=col, format='D')
c5=fits.Column(name="CH_05", array=col, format='D')
c6=fits.Column(name="CH_06", array=col, format='D')
c7=fits.Column(name="CH_07", array=col, format='D')
c8=fits.Column(name="CH_08", array=col, format='D')
c9=fits.Column(name="CH_09", array=col, format='D')
c10=fits.Column(name="CH_10", array=col, format='D')
c11=fits.Column(name="CH_11", array=col, format='D')
c12=fits.Column(name="CH_12", array=col, format='D')
c13=fits.Column(name="CH_13", array=col, format='D')
c14=fits.Column(name="CH_14", array=col, format='D')
c15=fits.Column(name="CH_15", array=col, format='D')
c16=fits.Column(name="CH_16", array=col, format='D')
c17=fits.Column(name="CH_17", array=col, format='D')
c18=fits.Column(name="CH_18", array=col, format='D')
c19=fits.Column(name="CH_19", array=col, format='D')
c20=fits.Column(name="CH_20", array=col, format='D')
c21=fits.Column(name="CH_21", array=col, format='D')
c22=fits.Column(name="CH_22", array=col, format='D')
c23=fits.Column(name="CH_23", array=col, format='D')
c24=fits.Column(name="CH_24", array=col, format='D')
c25=fits.Column(name="CH_25", array=col, format='D')
c26=fits.Column(name="CH_26", array=col, format='D')
c27=fits.Column(name="CH_27", array=col, format='D')
c28=fits.Column(name="CH_28", array=col, format='D')
c29=fits.Column(name="CH_29", array=col, format='D')
c30=fits.Column(name="CH_30", array=col, format='D')
c31=fits.Column(name="CH_31", array=col, format='D')

t = fits.BinTableHDU.from_columns([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31])
t.writeto('table2.fits',overwrite=True)

hdul = fits.open('table2.fits')
hdul.info()



c=np.array((col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col))
print(np.shape(c))

ct=c
#ct=np.transpose(c)
print(np.shape(ct),N)
for j in range(N):
    for i in range(Nmult[j]):
        #print('for j',j,' and i',i)
        #print(TRG[j,i],ADC[j,i])
        #print('en el canal',int(TRG[j,i]),'va el numero',ADC[j,i])
        #print('el canal',int(TRG[j,i]),'se corresponde con',)
        #print('antes habia',c[int(TRG[j,i]),j])
        ct[int(TRG[j,i]),j]=ADC[j,i]
        #print('ahora hay',c[int(TRG[j,i]),j])


c0=fits.Column(name="CH_00", array=ct[:,0], format='D')
c1=fits.Column(name="CH_01", array=ct[:,1], format='D')
c2=fits.Column(name="CH_02", array=ct[:,2], format='D')
c3=fits.Column(name="CH_03", array=ct[:,3], format='D')
c4=fits.Column(name="CH_04", array=ct[:,4], format='D')
c5=fits.Column(name="CH_05", array=ct[:,5], format='D')
c6=fits.Column(name="CH_06", array=ct[:,6], format='D')
c7=fits.Column(name="CH_07", array=ct[:,7], format='D')
c8=fits.Column(name="CH_08", array=ct[:,8], format='D')
c9=fits.Column(name="CH_09", array=ct[:,9], format='D')
c10=fits.Column(name="CH_10", array=ct[:,10], format='D')
c11=fits.Column(name="CH_11", array=ct[:,11], format='D')
c12=fits.Column(name="CH_12", array=ct[:,12], format='D')
c13=fits.Column(name="CH_13", array=ct[:,13], format='D')
c14=fits.Column(name="CH_14", array=ct[:,14], format='D')
c15=fits.Column(name="CH_15", array=ct[:,15], format='D')
c16=fits.Column(name="CH_16", array=ct[:,16], format='D')
c17=fits.Column(name="CH_17", array=ct[:,17], format='D')
c18=fits.Column(name="CH_18", array=ct[:,18], format='D')
c19=fits.Column(name="CH_19", array=ct[:,19], format='D')
c20=fits.Column(name="CH_20", array=ct[:,20], format='D')
c21=fits.Column(name="CH_21", array=ct[:,21], format='D')
c22=fits.Column(name="CH_22", array=ct[:,22], format='D')
c23=fits.Column(name="CH_23", array=ct[:,23], format='D')
c24=fits.Column(name="CH_24", array=ct[:,24], format='D')
c25=fits.Column(name="CH_25", array=ct[:,25], format='D')
c26=fits.Column(name="CH_26", array=ct[:,26], format='D')
c27=fits.Column(name="CH_27", array=ct[:,27], format='D')
c28=fits.Column(name="CH_28", array=ct[:,28], format='D')
c29=fits.Column(name="CH_29", array=ct[:,29], format='D')
c30=fits.Column(name="CH_30", array=ct[:,30], format='D')
c31=fits.Column(name="CH_31", array=ct[:,31], format='D')


t2 = fits.BinTableHDU.from_columns([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31])
t2.writeto('tablef.fits',overwrite=True)
