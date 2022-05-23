from astropy.io import fits
import dask.array as da
import numpy as np

fitsfile='Test_20220523_150848_LV0d5.fits'
#fitsfile='Prova_short.fits'


hdulist = fits.open(fitsfile,memmap=True,mode='denywrite')
intable = hdulist[1].data
hdulist.close()

NMULT = intable.field("NMULT")
N=np.size(NMULT)




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

print(np.shape(ADC))

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
#    cols[:,i]=fits.Column(name='test', array=col, format='F')
#    #cols[i] = fits.Column(name="CH_{:02d}".format(i), array=col, format='F')

#t = fits.BinTableHDU.from_columns(cols[0:31])



c=np.array((col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col,col))

del col 

at=np.copy(np.transpose(c))
bt=np.copy(np.transpose(c))
ct=np.copy(np.transpose(c))
dt=np.copy(np.transpose(c))

del c
# j=0,...,93
# i=0,...6


#xa = da.from_array(at, chunks=(10, 10))


for j in range(N):

    for i in range(NMULT[j]):

        if QUADID[j]==0:
            at[j,int(TRG[j,i])]=ADC[j,i]
            #print('Q ',QUADID[j],' CH ',int(TRG[j,i]),' EV',at[j,int(TRG[j,i])],' ADC',ADC[j,i])
        elif QUADID[j]==1:
            bt[j,int(TRG[j,i])]=ADC[j,i]
            #print('Q ',QUADID[j],' CH ',int(TRG[j,i]),' EV',bt[j,int(TRG[j,i])],' ADC',ADC[j,i])
        elif QUADID[j]==2:
            ct[j,int(TRG[j,i])]=ADC[j,i]
            #print('Q ',QUADID[j],' CH ',int(TRG[j,i]),' EV',ct[j,int(TRG[j,i])],' ADC',ADC[j,i])
        elif QUADID[j]==3:
            dt[j,int(TRG[j,i])]=ADC[j,i]
            #print('Q ',QUADID[j],' CH ',int(TRG[j,i]),' EV',dt[j,int(TRG[j,i])],' ADC',ADC[j,i])


del TRG,ADC,QUADID,NMULT

prim = fits.PrimaryHDU()

#atest=np.empty((N,32))
#
#print('atest',np.shape(atest))
#for i in range(32):
#
#    atest[:,i]=fits.Column(name="CH_{:02d}".format(i), array=at[:,0], format='F')
#
#ta=fits.BinTableHDU.from_columns([atest[:,0:32]])

a0=fits.Column(name="CH_00", array=at[:,0], format='F')
a1=fits.Column(name="CH_01", array=at[:,1], format='F')
a2=fits.Column(name="CH_02", array=at[:,2], format='F')
a3=fits.Column(name="CH_03", array=at[:,3], format='F')
a4=fits.Column(name="CH_04", array=at[:,4], format='F')
a5=fits.Column(name="CH_05", array=at[:,5], format='F')
a6=fits.Column(name="CH_06", array=at[:,6], format='F')
a7=fits.Column(name="CH_07", array=at[:,7], format='F')
a8=fits.Column(name="CH_08", array=at[:,8], format='F')
a9=fits.Column(name="CH_09", array=at[:,9], format='F')
a10=fits.Column(name="CH_10", array=at[:,10], format='F')
a11=fits.Column(name="CH_11", array=at[:,11], format='F')
a12=fits.Column(name="CH_12", array=at[:,12], format='F')
a13=fits.Column(name="CH_13", array=at[:,13], format='F')
a14=fits.Column(name="CH_14", array=at[:,14], format='F')
a15=fits.Column(name="CH_15", array=at[:,15], format='F')
a16=fits.Column(name="CH_16", array=at[:,16], format='F')
a17=fits.Column(name="CH_17", array=at[:,17], format='F')
a18=fits.Column(name="CH_18", array=at[:,18], format='F')
a19=fits.Column(name="CH_19", array=at[:,19], format='F')
a20=fits.Column(name="CH_20", array=at[:,20], format='F')
a21=fits.Column(name="CH_21", array=at[:,21], format='F')
a22=fits.Column(name="CH_22", array=at[:,22], format='F')
a23=fits.Column(name="CH_23", array=at[:,23], format='F')
a24=fits.Column(name="CH_24", array=at[:,24], format='F')
a25=fits.Column(name="CH_25", array=at[:,25], format='F')
a26=fits.Column(name="CH_26", array=at[:,26], format='F')
a27=fits.Column(name="CH_27", array=at[:,27], format='F')
a28=fits.Column(name="CH_28", array=at[:,28], format='F')
a29=fits.Column(name="CH_29", array=at[:,29], format='F')
a30=fits.Column(name="CH_30", array=at[:,30], format='F')
a31=fits.Column(name="CH_31", array=at[:,31], format='F')


ta = fits.BinTableHDU.from_columns([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31])

del a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31



b0=fits.Column(name="CH_00", array=bt[:,0], format='F')
b1=fits.Column(name="CH_01", array=bt[:,1], format='F')
b2=fits.Column(name="CH_02", array=bt[:,2], format='F')
b3=fits.Column(name="CH_03", array=bt[:,3], format='F')
b4=fits.Column(name="CH_04", array=bt[:,4], format='F')
b5=fits.Column(name="CH_05", array=bt[:,5], format='F')
b6=fits.Column(name="CH_06", array=bt[:,6], format='F')
b7=fits.Column(name="CH_07", array=bt[:,7], format='F')
b8=fits.Column(name="CH_08", array=bt[:,8], format='F')
b9=fits.Column(name="CH_09", array=bt[:,9], format='F')
b10=fits.Column(name="CH_10", array=bt[:,10], format='F')
b11=fits.Column(name="CH_11", array=bt[:,11], format='F')
b12=fits.Column(name="CH_12", array=bt[:,12], format='F')
b13=fits.Column(name="CH_13", array=bt[:,13], format='F')
b14=fits.Column(name="CH_14", array=bt[:,14], format='F')
b15=fits.Column(name="CH_15", array=bt[:,15], format='F')
b16=fits.Column(name="CH_16", array=bt[:,16], format='F')
b17=fits.Column(name="CH_17", array=bt[:,17], format='F')
b18=fits.Column(name="CH_18", array=bt[:,18], format='F')
b19=fits.Column(name="CH_19", array=bt[:,19], format='F')
b20=fits.Column(name="CH_20", array=bt[:,20], format='F')
b21=fits.Column(name="CH_21", array=bt[:,21], format='F')
b22=fits.Column(name="CH_22", array=bt[:,22], format='F')
b23=fits.Column(name="CH_23", array=bt[:,23], format='F')
b24=fits.Column(name="CH_24", array=bt[:,24], format='F')
b25=fits.Column(name="CH_25", array=bt[:,25], format='F')
b26=fits.Column(name="CH_26", array=bt[:,26], format='F')
b27=fits.Column(name="CH_27", array=bt[:,27], format='F')
b28=fits.Column(name="CH_28", array=bt[:,28], format='F')
b29=fits.Column(name="CH_29", array=bt[:,29], format='F')
b30=fits.Column(name="CH_30", array=bt[:,30], format='F')
b31=fits.Column(name="CH_31", array=bt[:,31], format='F')


tb = fits.BinTableHDU.from_columns([b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31])

del b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31

c0=fits.Column(name="CH_00", array=ct[:,0], format='F')
c1=fits.Column(name="CH_01", array=ct[:,1], format='F')
c2=fits.Column(name="CH_02", array=ct[:,2], format='F')
c3=fits.Column(name="CH_03", array=ct[:,3], format='F')
c4=fits.Column(name="CH_04", array=ct[:,4], format='F')
c5=fits.Column(name="CH_05", array=ct[:,5], format='F')
c6=fits.Column(name="CH_06", array=ct[:,6], format='F')
c7=fits.Column(name="CH_07", array=ct[:,7], format='F')
c8=fits.Column(name="CH_08", array=ct[:,8], format='F')
c9=fits.Column(name="CH_09", array=ct[:,9], format='F')
c10=fits.Column(name="CH_10", array=ct[:,10], format='F')
c11=fits.Column(name="CH_11", array=ct[:,11], format='F')
c12=fits.Column(name="CH_12", array=ct[:,12], format='F')
c13=fits.Column(name="CH_13", array=ct[:,13], format='F')
c14=fits.Column(name="CH_14", array=ct[:,14], format='F')
c15=fits.Column(name="CH_15", array=ct[:,15], format='F')
c16=fits.Column(name="CH_16", array=ct[:,16], format='F')
c17=fits.Column(name="CH_17", array=ct[:,17], format='F')
c18=fits.Column(name="CH_18", array=ct[:,18], format='F')
c19=fits.Column(name="CH_19", array=ct[:,19], format='F')
c20=fits.Column(name="CH_20", array=ct[:,20], format='F')
c21=fits.Column(name="CH_21", array=ct[:,21], format='F')
c22=fits.Column(name="CH_22", array=ct[:,22], format='F')
c23=fits.Column(name="CH_23", array=ct[:,23], format='F')
c24=fits.Column(name="CH_24", array=ct[:,24], format='F')
c25=fits.Column(name="CH_25", array=ct[:,25], format='F')
c26=fits.Column(name="CH_26", array=ct[:,26], format='F')
c27=fits.Column(name="CH_27", array=ct[:,27], format='F')
c28=fits.Column(name="CH_28", array=ct[:,28], format='F')
c29=fits.Column(name="CH_29", array=ct[:,29], format='F')
c30=fits.Column(name="CH_30", array=ct[:,30], format='F')
c31=fits.Column(name="CH_31", array=ct[:,31], format='F')



tc = fits.BinTableHDU.from_columns([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31])

del c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31

d0=fits.Column(name="CH_00", array=dt[:,0], format='F')
d1=fits.Column(name="CH_01", array=dt[:,1], format='F')
d2=fits.Column(name="CH_02", array=dt[:,2], format='F')
d3=fits.Column(name="CH_03", array=dt[:,3], format='F')
d4=fits.Column(name="CH_04", array=dt[:,4], format='F')
d5=fits.Column(name="CH_05", array=dt[:,5], format='F')
d6=fits.Column(name="CH_06", array=dt[:,6], format='F')
d7=fits.Column(name="CH_07", array=dt[:,7], format='F')
d8=fits.Column(name="CH_08", array=dt[:,8], format='F')
d9=fits.Column(name="CH_09", array=dt[:,9], format='F')
d10=fits.Column(name="CH_10", array=dt[:,10], format='F')
d11=fits.Column(name="CH_11", array=dt[:,11], format='F')
d12=fits.Column(name="CH_12", array=dt[:,12], format='F')
d13=fits.Column(name="CH_13", array=dt[:,13], format='F')
d14=fits.Column(name="CH_14", array=dt[:,14], format='F')
d15=fits.Column(name="CH_15", array=dt[:,15], format='F')
d16=fits.Column(name="CH_16", array=dt[:,16], format='F')
d17=fits.Column(name="CH_17", array=dt[:,17], format='F')
d18=fits.Column(name="CH_18", array=dt[:,18], format='F')
d19=fits.Column(name="CH_19", array=dt[:,19], format='F')
d20=fits.Column(name="CH_20", array=dt[:,20], format='F')
d21=fits.Column(name="CH_21", array=dt[:,21], format='F')
d22=fits.Column(name="CH_22", array=dt[:,22], format='F')
d23=fits.Column(name="CH_23", array=dt[:,23], format='F')
d24=fits.Column(name="CH_24", array=dt[:,24], format='F')
d25=fits.Column(name="CH_25", array=dt[:,25], format='F')
d26=fits.Column(name="CH_26", array=dt[:,26], format='F')
d27=fits.Column(name="CH_27", array=dt[:,27], format='F')
d28=fits.Column(name="CH_28", array=dt[:,28], format='F')
d29=fits.Column(name="CH_29", array=dt[:,29], format='F')
d30=fits.Column(name="CH_30", array=dt[:,30], format='F')
d31=fits.Column(name="CH_31", array=dt[:,31], format='F')

td = fits.BinTableHDU.from_columns([d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31])

del d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31



infitsf= fits.HDUList([prim,ta,tb,tc,td])
infitsf.writeto('Test_20220523_150848.fits',overwrite=True)

del ta,tb,tc,td
infitsf.close()
#