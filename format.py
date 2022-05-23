from astropy.io import fits

import numpy as np

fitsfile='Fe55v5_short.fits'

hdulist = fits.open(fitsfile)
intable = hdulist[1].data
hdulist.close()

Nmult = intable.field("NMULT")
N=np.size(Nmult)




QUADID= intable.field("QUADID")

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

col=np.ones(N)
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



c=np.array((col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col))


at=np.transpose(c)
bt=np.transpose(c)
ct=np.transpose(c)
dt=np.transpose(c)


for j in range(N):
    for i in range(Nmult[j]):
        if QUADID[j]==0:
            at[j,int(TRG[j,i])]=ADC[j,i]
        elif QUADID[j]==1:
            bt[j,int(TRG[j,i])]=ADC[j,i]
        elif QUADID[j]==1:
            ct[j,int(TRG[j,i])]=ADC[j,i]
        elif QUADID[j]==1:
            dt[j,int(TRG[j,i])]=ADC[j,i]
    

a0=fits.Column(name="CH_00", array=at[:,0], format='D')
a1=fits.Column(name="CH_01", array=at[:,1], format='D')
a2=fits.Column(name="CH_02", array=at[:,2], format='D')
a3=fits.Column(name="CH_03", array=at[:,3], format='D')
a4=fits.Column(name="CH_04", array=at[:,4], format='D')
a5=fits.Column(name="CH_05", array=at[:,5], format='D')
a6=fits.Column(name="CH_06", array=at[:,6], format='D')
a7=fits.Column(name="CH_07", array=at[:,7], format='D')
a8=fits.Column(name="CH_08", array=at[:,8], format='D')
a9=fits.Column(name="CH_09", array=at[:,9], format='D')
a10=fits.Column(name="CH_10", array=at[:,10], format='D')
a11=fits.Column(name="CH_11", array=at[:,11], format='D')
a12=fits.Column(name="CH_12", array=at[:,12], format='D')
a13=fits.Column(name="CH_13", array=at[:,13], format='D')
a14=fits.Column(name="CH_14", array=at[:,14], format='D')
a15=fits.Column(name="CH_15", array=at[:,15], format='D')
a16=fits.Column(name="CH_16", array=at[:,16], format='D')
a17=fits.Column(name="CH_17", array=at[:,17], format='D')
a18=fits.Column(name="CH_18", array=at[:,18], format='D')
a19=fits.Column(name="CH_19", array=at[:,19], format='D')
a20=fits.Column(name="CH_20", array=at[:,20], format='D')
a21=fits.Column(name="CH_21", array=at[:,21], format='D')
a22=fits.Column(name="CH_22", array=at[:,22], format='D')
a23=fits.Column(name="CH_23", array=at[:,23], format='D')
a24=fits.Column(name="CH_24", array=at[:,24], format='D')
a25=fits.Column(name="CH_25", array=at[:,25], format='D')
a26=fits.Column(name="CH_26", array=at[:,26], format='D')
a27=fits.Column(name="CH_27", array=at[:,27], format='D')
a28=fits.Column(name="CH_28", array=at[:,28], format='D')
a29=fits.Column(name="CH_29", array=at[:,29], format='D')
a30=fits.Column(name="CH_30", array=at[:,30], format='D')
a31=fits.Column(name="CH_31", array=at[:,31], format='D')


b0=fits.Column(name="CH_00", array=bt[:,0], format='D')
b1=fits.Column(name="CH_01", array=bt[:,1], format='D')
b2=fits.Column(name="CH_02", array=bt[:,2], format='D')
b3=fits.Column(name="CH_03", array=bt[:,3], format='D')
b4=fits.Column(name="CH_04", array=bt[:,4], format='D')
b5=fits.Column(name="CH_05", array=bt[:,5], format='D')
b6=fits.Column(name="CH_06", array=bt[:,6], format='D')
b7=fits.Column(name="CH_07", array=bt[:,7], format='D')
b8=fits.Column(name="CH_08", array=bt[:,8], format='D')
b9=fits.Column(name="CH_09", array=bt[:,9], format='D')
b10=fits.Column(name="CH_10", array=bt[:,10], format='D')
b11=fits.Column(name="CH_11", array=bt[:,11], format='D')
b12=fits.Column(name="CH_12", array=bt[:,12], format='D')
b13=fits.Column(name="CH_13", array=bt[:,13], format='D')
b14=fits.Column(name="CH_14", array=bt[:,14], format='D')
b15=fits.Column(name="CH_15", array=bt[:,15], format='D')
b16=fits.Column(name="CH_16", array=bt[:,16], format='D')
b17=fits.Column(name="CH_17", array=bt[:,17], format='D')
b18=fits.Column(name="CH_18", array=bt[:,18], format='D')
b19=fits.Column(name="CH_19", array=bt[:,19], format='D')
b20=fits.Column(name="CH_20", array=bt[:,20], format='D')
b21=fits.Column(name="CH_21", array=bt[:,21], format='D')
b22=fits.Column(name="CH_22", array=bt[:,22], format='D')
b23=fits.Column(name="CH_23", array=bt[:,23], format='D')
b24=fits.Column(name="CH_24", array=bt[:,24], format='D')
b25=fits.Column(name="CH_25", array=bt[:,25], format='D')
b26=fits.Column(name="CH_26", array=bt[:,26], format='D')
b27=fits.Column(name="CH_27", array=bt[:,27], format='D')
b28=fits.Column(name="CH_28", array=bt[:,28], format='D')
b29=fits.Column(name="CH_29", array=bt[:,29], format='D')
b30=fits.Column(name="CH_30", array=bt[:,30], format='D')
b31=fits.Column(name="CH_31", array=bt[:,31], format='D')

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

d0=fits.Column(name="CH_00", array=dt[:,0], format='D')
d1=fits.Column(name="CH_01", array=dt[:,1], format='D')
d2=fits.Column(name="CH_02", array=dt[:,2], format='D')
d3=fits.Column(name="CH_03", array=dt[:,3], format='D')
d4=fits.Column(name="CH_04", array=dt[:,4], format='D')
d5=fits.Column(name="CH_05", array=dt[:,5], format='D')
d6=fits.Column(name="CH_06", array=dt[:,6], format='D')
d7=fits.Column(name="CH_07", array=dt[:,7], format='D')
d8=fits.Column(name="CH_08", array=dt[:,8], format='D')
d9=fits.Column(name="CH_09", array=dt[:,9], format='D')
d10=fits.Column(name="CH_10", array=dt[:,10], format='D')
d11=fits.Column(name="CH_11", array=dt[:,11], format='D')
d12=fits.Column(name="CH_12", array=dt[:,12], format='D')
d13=fits.Column(name="CH_13", array=dt[:,13], format='D')
d14=fits.Column(name="CH_14", array=dt[:,14], format='D')
d15=fits.Column(name="CH_15", array=dt[:,15], format='D')
d16=fits.Column(name="CH_16", array=dt[:,16], format='D')
d17=fits.Column(name="CH_17", array=dt[:,17], format='D')
d18=fits.Column(name="CH_18", array=dt[:,18], format='D')
d19=fits.Column(name="CH_19", array=dt[:,19], format='D')
d20=fits.Column(name="CH_20", array=dt[:,20], format='D')
d21=fits.Column(name="CH_21", array=dt[:,21], format='D')
d22=fits.Column(name="CH_22", array=dt[:,22], format='D')
d23=fits.Column(name="CH_23", array=dt[:,23], format='D')
d24=fits.Column(name="CH_24", array=dt[:,24], format='D')
d25=fits.Column(name="CH_25", array=dt[:,25], format='D')
d26=fits.Column(name="CH_26", array=dt[:,26], format='D')
d27=fits.Column(name="CH_27", array=dt[:,27], format='D')
d28=fits.Column(name="CH_28", array=dt[:,28], format='D')
d29=fits.Column(name="CH_29", array=dt[:,29], format='D')
d30=fits.Column(name="CH_30", array=dt[:,30], format='D')
d31=fits.Column(name="CH_31", array=dt[:,31], format='D')



prim = fits.PrimaryHDU()

taa = fits.BinTableHDU.from_columns([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31])

tbb = fits.BinTableHDU.from_columns([b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31])

tcc = fits.BinTableHDU.from_columns([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31])

tdd = fits.BinTableHDU.from_columns([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31])


infitsf= fits.HDUList([prim,taa,tbb,tcc,tdd])

infitsf.writeto('tablef.fits',overwrite=True)
