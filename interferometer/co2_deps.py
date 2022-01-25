#!/usr/bin/env python
# coding: utf-8

#+#LFM classification dependencies
#+This file contains useful functions commonly used in the ML notebooks
#+***
from __future__ import print_function
import h5py as h5
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from scipy.fft import fft
from astropy.convolution import convolve, Box1DKernel
from matplotlib import ticker, cm, colors
import tensorflow.compat.v2 as tf


def frame_stacking(I,frwin=[0,0,'0']):
    if frwin[0]+frwin[1]==0:
        return I
    else:
        if frwin[2]=='m':
            B=np.hamming(2*frwin[0]+1)[:frwin[0]]
            A=np.hamming(2*frwin[1]+1)[frwin[1]+1:]
        elif frwin[2]=='n':
            B=np.hanning(2*frwin[0]+1)[:frwin[0]]
            A=np.hanning(2*frwin[1]+1)[frwin[1]+1:]
        else:
            B=np.ones(frwin[0])
            A=np.ones(frwin[1])
        n_fr,n_fe=I.shape
        tmp=np.ones((n_fr+frwin[0]+frwin[1],n_fe))
        tmp[:frwin[0],:]=I[0,:]
        if frwin[1]!=0:
            tmp[-frwin[1]:,:]=I[-1,:]
        tmp[frwin[0]:frwin[0]+n_fr,:]=I
        t_f_w=[]
        for i in range(frwin[0],frwin[0]+n_fr):
            tt=(tmp[i-frwin[0]:i,:].T*B).T
            bef=np.reshape(tt,(1,-1))[0]
            tt=(tmp[i+1:i+frwin[1]+1,:].T*A).T
            aft=np.reshape(tt,(1,-1))[0]
            t_f_w.append(np.concatenate((bef,tmp[i,:],aft),0))
        t_f_w=np.array(t_f_w)
        return t_f_w


def save_targets(labels, fname, path="/projects/EKOLEMEN/agarcia/df1/"):
    # labels is a list of length shots, where each element is an array of [classes, time]
    # fname would be the file ID
    with open(path + f'targets_{fname}.txt'  , "wb") as fp:
        pickle.dump(labels, fp)
    return


def save_targets_shots(labels, fname, path="/projects/EKOLEMEN/agarcia/df1/"):
    # labels is a list of length shots, where each element is an array of [classes, time]
    # fname would be the file ID
    with open(path + f'targets_shots_{fname}.txt'  , "wb") as fp:
        pickle.dump(labels, fp)
    return


def describe(dataframe):
    return dataframe.describe().iloc[[0,3,7]].astype(int)


def plot_flag_distribution(df0, df1, df2, flag_labels, order=1):
    data = np.asarray([df0.sum(0)[2:], df1.sum(0)[2:], df2.sum(0)[2:]])*order

    fig, ax = plt.subplots(1,3,figsize=(12,4))
    ax = ax.flatten()
    for i,d in enumerate(data):
        ax[i].bar(flag_labels, d)
    ax[1].set_xlabel('Modes')
    ax[0].set_ylabel('Counts')
                   
    plt.show()
    return


def load_data(fdir):
    with open(fdir, 'rb') as fp:
        loaded_data = pickle.load(fp)
    return loaded_data


def write_data(fdir, data):
    with open(fdir, 'wb') as fp:
        pickle.dump(data, fp)
    return 


def load_time_series(fid, full=False, chord1='v2', chord2='r0'):
    signal1 = h5.File(f'/projects/EKOLEMEN/agarcia/time_series/v2r0/signal1_{fid}.h5', 'r')[f'dp1{chord1}uf'][()]
    signal2 = h5.File(f'/projects/EKOLEMEN/agarcia/time_series/v2r0/signal2_{fid}.h5', 'r')[f'dp1{chord2}uf'][()]
    shots = h5.File(f'/projects/EKOLEMEN/agarcia/time_series/shots_{fid}.h5', 'r')['shot'][()]
    # Since targets are written from (0,2) s, need to cut tsginal
    if full:
        ind = np.ones_like(signal1[0], dtype=bool) #don't cut off at 0 and 2000 ms
    else:
        ind = np.load('/projects/EKOLEMEN/agarcia/time_series/tsignal_2000.npy')
    return signal1[:,ind], signal2[:,ind], shots[:,0].astype(int)

def load_time_series_tensor(fid, chord1, chord2):
    tdir = '/projects/EKOLEMEN/agarcia/time_series'
    
    if chord1=='v1' and chord2=='v3':
        signal1 = h5.File(f'{tdir}/v1v3/signal1_{fid}.h5', 'r')[f'dp1{chord1}uf'][()]
        signal2 = h5.File(f'{tdir}/v1v3/signal2_{fid}.h5', 'r')[f'dp1{chord2}uf'][()]
    if chord1=='v2' and chord2=='r0':
        signal1 = h5.File(f'{tdir}/v2r0/signal1_{fid}.h5', 'r')[f'dp1{chord1}uf'][()]
        signal2 = h5.File(f'{tdir}/v2r0/signal2_{fid}.h5', 'r')[f'dp1{chord2}uf'][()]
    
    shots = h5.File(f'{tdir}/shots_{fid}.h5', 'r')['shot'][()]
    
    return signal1, signal2, np.asarray(shots[:,0].astype(int))


def get_fid(shotnum, shotlist):
    return shotlist[shotlist>=shotnum].min()


def ae_co2(sig_in1, sig_in2, time, nff=8192, tmin=0, tmax=2000):
    # sig_in1 is v2
    # sig_in2 is r0
    #time is the absicca of the data
    #tmin, tmax are define the FFT limits
    # nff is num fft filters 
    nts = len(time)
    dtt = (time[-1]-time[0])/(nts-1)
    overlap = 80
    n0 = np.argmin(abs(tmin-time))
    noavgremove = 0
    power = 1
    fsmot = 1
    crosspower = True
    frmin = 20
    frmax = 250

    nwin = int(1/(1-overlap*1e-2)*(tmax-tmin)/(dtt*nff)+1)

    #added this as a quick fix 2/10/10
    ttemps = np.zeros(nwin)
    n0t = 0
    for i in range(nwin):
        ttemps[i] = tmin + n0t * dtt
        n0t = n0t + nff*(1-overlap*1e-2)

    nww = np.argmin(abs(ttemps+dtt*nff/2-tmax))
    nwin = nww+1

    # Initialize
    ampsp = np.zeros((nwin, int(nff/2+1)))
    times = np.zeros(nwin)
    datdetrend = np.zeros((3, nwin)) #mean, yint, slope

    detrend = 'linear' #Doing linear detrending
    window = 'hann' # Using Hanning window


    # --- Calculate crosspower ---


    sig  =  sig_in2.copy()
    sigb  =  sig_in1.copy()
    for i in range(nwin):
        isig0 = int(n0-nff/2)
        isigf = int(n0+nff/2-1) ### -1 makes the length 8191 but w/e it's only one poing

        sig2 = sig[isig0:isigf]

        if noavgremove <= 0:
            sig2 = sig2-np.mean(sig2)  #removing average


        #*********** Detrending w/ linear detrending if requested ***********
        if detrend:
            tshor = time[isig0:isigf]
            adt = np.polyfit(tshor,sig2,1) #Coefficients in decreasing powers
            dtfit = np.polyval(adt, tshor)
            datdetrend[1::,i] = np.flip(adt) #Flip to match idl code
            datdetrend[0,i] = np.mean(sig2)
            sig2 = sig2-dtfit

        if len(sigb)>1:
            sig2b = sigb[isig0:isigf]

            if noavgremove <= 0:
                sig2b = sig2b-np.mean(sig2b)  #removing average

            if detrend:   #detrend or not
                adt = np.polyfit(tshor,sig2b,1)
                dtfitb = np.polyval(adt, tshor)
                sig2b = sig2b-dtfitb
        #******************************************************************
            sig2b = sig2b*np.hanning(len(sig2b))

        sig2 = sig2*np.hanning(len(sig2))
        #******************************************************************


        tsig2 = time[isig0:isigf]  #time of window is just average of timebase

        t0 = np.mean(tsig2)


        if i == 0: typp = 'AutoAmp'
        if i == 0 and power != 0: typp = 'AutoPower w/ units'
        #Using fftf routine to get frequency base
        #****************************************************
        fftn, fo = fftf(tsig2,sig2,hann=0) #set hann = 0 because sig2 will already use hann
        fftn = abs(fftn)
        nsmot = (fsmot/(fo[1]-fo[0])).astype(int)  #checking to see how many to smooth over

        if power != 0:    #If requesting actual power in real units
            pow = fft2ps(fftn,sig2,dtt)#
            pow = abs(pow)


        fftn = abs(fftn)
        #Below conditional is close but not exact, not sure on the functions
        # or mode kwarg
        if nsmot>2:
            fftn = convolve(fftn, Box1DKernel(nsmot, mode='center'))
            if power != 0: pow = convolve(pow, Box1DKernel(nsmot, mode='center'))

        times[i] = t0


        n0 = n0+nff*(1.-overlap*.01)


        ampsp[i,:] = fftn[:int(nff/2)+1]*2.**.5 #added 2.^.5 #not sure if this is right though should be 2?
        ###algo
        if power != 0: ampsp[i,:] = pow[:int(nff/2)+1]*2.      #will be off if hanning is used
        ###algo

        #****************************************************
        #crosspower calculation
        #****************************************************
        if crosspower:
            if i == 0: typp = 'Crosspower'

            fftc1 = fft(sig2, norm='forward')
            fftc2 = fft(sig2b, norm='forward')
            cpower = abs(fftc1*np.conj(fftc2))
            if nsmot>2: cpower = convolve(cpower, Box1DKernel(nsmot))

            #Formally this is not correct it should be the expression below
            #I think this amounts to just |power1|*|power2|
            #if no freq. smoothing is used.
            #cpower = abs(smooth(fftc1*conj(fftc2),nsmot,/edge_truncate,/nan))


            ampsp[i,:] = cpower[:int(nff/2)+1]*2.
            ###algo

    ampsp = ampsp[:, (fo>=frmin)&(fo<=frmax)]
    fo = fo[(fo>=frmin)&(fo<=frmax)]
    #Remove final points to make image rectangular
    return ampsp[:-1], fo, times[:-1]


def rebin(a, fr, ti, freq_factor=10, times_factor=8):
    # Max pooling
    # freq_factor must be a power of 2
    # times_factor must be a power of 2

    nfreq = int((len(fr))/freq_factor)
    ntimes = int((len(ti))/times_factor)
    shape = (ntimes,nfreq)
    
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).max(-1).max(1)


def fftf(t,s,hann=0):
    #11/13/15 added return of complex fft
    nnf=len(t)

    dt=(t[-1]-t[0]) / (nnf-1.)
    fo = np.arange(int(nnf/2.+2.)) / (nnf*dt)


    if hann==1:
        ss=s*np.hanning(len(s))
        fftc=fft(ss)
        fftn=abs(fftc)
    else:
        fftc=fft(s, norm='forward')
        fftn=abs(fftc)
        
    return fftn, fo


#This program takes in an array of fft values
#dt = timestep.  if passed in ms, 
#and signal in V then units of 
#power spectrum is V**2/kHz

#fftsig is a fft computed somewhere else. 
#sig is the original signal the fft is computed of
#if a window was used don't send the window*sig
#otherwise the integration won't work out
#sends back same dimensions as fftsig

def fft2ps(fftsig, sig, dt):
    #print,'Make sure you subtract off average before doing fft'
    #print, 'Also,definitely do not put a hanning filter before the average subtraction'
    sig2ps = sig #-np.mean(sig)
    fftsig2 = fftsig
    #fftsig2(0) = fftsig2(0)*0.
    nt = len(sig)
    a2 = np.mean(sig2ps**2.)

    dff = 1./nt/dt

    c1 = a2/dff/np.sum(abs(fftsig2)**2.)

    #checks to see if it is including negative frequency components already
    if len(fftsig) == len(sig): c1 = 2.*c1   

    return c1*abs(fftsig2)**2.


def get_colors(filename):
    #! Get IDL STD GAMMA-II color table by using commands
    #! IDL> device,decomposed=0
    #! IDL> loadct,5
    #! IDL> COMMON COLORS, R_orig, G_orig, B_orig, R_curr, G_curr, B_curr
    #! IDL> colors = {ro:R_orig, go:G_orig, bo:B_orig, rc:R_curr, gc:G_curr, bc:B_curr}
    #! IDL> fout = '/home/garciaav/colors.h5'
    #! IDL> write_hdf5,colors,filename=fout

    colors_file = h5.File(filename, 'r')
    ro = colors_file['ro'][()].tolist()
    go = colors_file['go'][()].tolist()
    bo = colors_file['bo'][()].tolist()
    N = len(ro)
    cmin = min((ro+go+bo))
    cmax = max((ro+go+bo))

    return [0] + np.linspace(1,cmax,(N-1)).tolist(), ro, go, bo


def make_colourmap(ind, red, green, blue, name):
    #! Reference: http://astrolitterbox.blogspot.com/2012/11/rewriting-idl-color-table-routines-in.html
    #! Reference: https://github.com/astrolitterbox/DataUtils/blob/master/califa_cmap.py
    newInd = range(0, 256)
    r = np.interp(newInd, ind, red, left=None, right=None)
    g = np.interp(newInd, ind, green, left=None, right=None)
    b = np.interp(newInd, ind, blue, left=None, right=None)
    colours = np.transpose(np.asarray((r, g, b)))
    fctab= colours/255.0
    cmap = colors.ListedColormap(fctab, name=name,N=None) 
    return cmap


def plot_images(images, time, frequency, shot, cmap=None, legend=False):
    fig, ax = plt.subplots(figsize=(16,8))

    #! Define levels for contour resolution
    locator = ticker.LogLocator(base=10, numticks=8)
    lev_exp = np.linspace(np.floor(np.log10(images.min())-1), 
                          np.ceil(np.log10(images.max())+1), 50)
    levs = np.power(10, lev_exp)

    c = ax.contourf(time, frequency, images.T, levs,locator=locator, cmap=cmap)
    if legend:
        cbar = fig.colorbar(c, format='%.e', ticks=locator)
        cbar.ax.tick_params(labelsize=18) 

    ax.set_title('Shot ' + str(shot), fontsize=18)
    ax.set_xlabel('Time [ms]', fontsize=18)
    ax.set_ylabel('Frequency [kHz]', fontsize=18)
    ax.set_ylim(5,200)
    ax.tick_params(axis='both', labelsize=18)

    plt.show()
    return


def plot_time_series(y1, y2, x, discharge, dataframe):
    # y1 is signal1 which is v2 chord
    # y2 is signal2 which is r0 chord
    # discharge is the shot number
    # dataframe used for windows, provide most general dataframe if possible
    cycler = ['b', 'c', 'm', 'k']
    shot_label = dataframe[dataframe['shot']==discharge] # extract the ae data from the labels
    plot_marks = shot_label.apply(lambda row: row[(row>=1) & (row<=4)].index, axis=1).values # name of the ae for plot

    
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(x,y1,label='v2',c='r')
    ax.plot(x,y2,label='r0',c='g')
    ax.set_ylabel('Signal', fontsize=18)
    ax.set_xlabel('Time [ms]', fontsize=18)
    ax.set_xlim(0, 2000)
    ax.tick_params(axis='both', labelsize=18)
    

    # Window
    for i in range(shot_label.shape[0]):
        t0 = shot_label['time'].iloc[i]
        tlo = t0 - 125 #ms
        thi = t0 + 125 #ms
        ax.axvline(tlo, c='k', linestyle='--')
        ax.axvline(thi, c='k', linestyle='--')
        
    # Flags and their annotation
    for tm,mrk in zip(shot_label['time'], plot_marks):
        ax.axvline(t0, c='k', linestyle=':')
        for i,m in enumerate(mrk):
            ax.annotate(m, (tm, np.amax([y1,y2])-(i*40)), color='k',
                        fontsize=18, ha='center', va='center')

    ax.set_title(f'Shot {discharge}', fontsize=18)
    plt.show()
    return

#+ Modified Metrics
class mod_TruePositives(tf.keras.metrics.Metric):

    def __init__(self, batch_size, name='mod_TP', nchunk=10, threshold=0.5, tlabel=200, **kwargs):
        #+ 'th'  Classification prediction threshold
        #+ 'Nw'  Number of chunks per shot
        #+ 'dtl' Time interval to "stretch" label
        #+ 'Ns'  Number of shots (batch_size and nchunk must be correct)    

        super(mod_TruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='mtp', initializer='zeros', dtype=tf.float32, shape=5)
        self.th = threshold
        self.Nw = nchunk
        self.dtl = tlabel
        self.Ns = batch_size // nchunk

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.float32) # (batch_size, classes)
        y_pred = tf.cast(y_pred, tf.float32) # (batch_size, classes)

        # Determine number of extra windows to consider
        dtw = 2000 // self.Nw
        Ne = max( int(self.dtl // dtw), 0 )
#         print(f'Ne, dtl, dtw = {Ne}, {self.dtl}, {dtw}\n')
        
        values = tf.cast([[0,0,0,0,0]], dtype=tf.bool) # Remove later
        for i in range(self.Ns):     # Loop over shots
            for j in range(self.Nw): # Loop over windows
#                 print(f'Shot {i}, win {j}')
                ic = i*self.Nw+j                      # Current window iteration 
                ilo = max(i*self.Nw, i*self.Nw+j-Ne)       # Don't check previous shot
                ihi = min((i+1)*self.Nw, i*self.Nw+j+Ne+1) # Don't check next shot

                v1_temp = tf.equal(y_true[ic:ic+1, :], 1.0)
                v2_temp = tf.reduce_any( tf.abs(y_true[ic:ic+1,:]-y_pred[ilo:ihi]) <= self.th, axis=0, keepdims=True)
                values_temp = tf.logical_and(v1_temp, v2_temp) 

#                 print(f'\t     True {(ic)}:{(ic+1)}')
#                 print(f'\t     Look {max(ic-j, ic-Ne)}:{min((i+1)*self.Nw, ic+1+Ne)}')
                values = tf.concat([values, values_temp], 0)
        values = values[1:] #Remove first entry
#         print(f'\nvalues = \n{values.shape}')
#         print(f'\nTP = {tf.reduce_sum(tf.cast(values, tf.float32), axis=0)}')
        self.true_positives.assign_add(tf.reduce_sum(tf.cast(values, tf.float32), axis=0))

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(tf.zeros(5))

        
        
class mod_FalsePositives(tf.keras.metrics.Metric):
    
    def __init__(self, batch_size=200, name='mod_FP', nchunk=10, threshold=0.5, tlabel=200, **kwargs):
        #+ 'th'  Classification prediction threshold
        #+ 'Nw'  Number of chunks per shot
        #+ 'dtl' Time interval to "stretch" label
        #+ 'Ns'  Number of shots (batch_size and nchunk must be correct)    

        super(mod_FalsePositives, self).__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name='mfp', initializer='zeros', dtype=tf.float32, shape=5)
        self.th = threshold
        self.Nw = nchunk
        self.dtl = tlabel
        self.Ns = batch_size // nchunk     

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.float32) # (batch_size, classes)
        y_pred = tf.cast(y_pred, tf.float32) # (batch_size, classes)

        # Determine number of extra windows to consider
        dtw = 2000 // self.Nw
        Ne = max( int(self.dtl // dtw), 0 )
#         print(f'Ne, dtl, dtw = {Ne}, {self.dtl}, {dtw}\n')

        values = tf.cast([[0,0,0,0,0]], dtype=tf.bool) # Remove later
        for i in range(self.Ns):     # Loop over shots
            for j in range(self.Nw): # Loop over windows
#                 print(f'Shot {i}, win {j}')
                ic = i*self.Nw+j                      # Current window iteration 
                ilo = max(i*self.Nw, i*self.Nw+j-Ne)       # Don't check previous shot
                ihi = min((i+1)*self.Nw, i*self.Nw+j+Ne+1) # Don't check next shot

                v1_temp = tf.equal(y_true[ic:ic+1, :], 0.0)
                v2_temp = tf.reduce_all( tf.abs(y_true[ic:ic+1,:]-y_pred[ilo:ihi,:]) > self.th, axis=0, keepdims=True)
                values_temp = tf.logical_and(v1_temp, v2_temp) 

#                 print(f'\t     True {(ic)}:{(ic+1)}')
#                 print(f'\t     Look {max(ic-j, ic-Ne)}:{min((i+1)*self.Nw, ic+1+Ne)}')
                values = tf.concat([values, values_temp], 0)
        values = values[1:] #Remove first entry
#         print(f'\nvalues = \n{values.shape}')
#         print(f'\FP = {tf.reduce_sum(tf.cast(values, tf.float32), axis=0)}')                
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(values, tf.float32), axis=0))
                
    def result(self):
        return self.false_positives

    def reset_states(self):
        self.false_positives.assign(tf.zeros(5))
        
        
        
class mod_FalseNegatives(tf.keras.metrics.Metric):

    def __init__(self, batch_size=200, name='mod_FN', threshold=0.5, **kwargs):
        super(mod_FalseNegatives, self).__init__(name=name, **kwargs)
        self.false_negatives = self.add_weight(name='mfn', initializer='zeros', dtype=tf.float32, shape=5)
        self.th = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        values_build = (tf.abs(y_true - y_pred) >= self.th)
        values_build = tf.logical_and(tf.equal(y_true, 1.0), values_build) 
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(values_build, tf.float32), axis=0))

    def result(self):
        return self.false_negatives

    def reset_states(self):
        self.false_negatives.assign(tf.zeros(5))

        
        
class mod_TrueNegatives(tf.keras.metrics.Metric):
    def __init__(self, batch_size=200, name='mod_TN', threshold=0.5, **kwargs):
        super(mod_TrueNegatives, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='mtn', initializer='zeros', dtype=tf.float32, shape=5)
        self.th = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        values_build = (tf.abs(y_true - y_pred) <= self.th)
        values_build = tf.logical_and(tf.equal(y_true, 0.0), values_build) 
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(values_build, tf.float32), axis=0))

    def result(self):
        return self.true_negatives

    def reset_states(self):
        self.true_negatives.assign(tf.zeros(5))