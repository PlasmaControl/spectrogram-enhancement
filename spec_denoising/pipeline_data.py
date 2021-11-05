'''
This code processes ECE raw ECE data to make spectrograms, run a denoising pipeline, and save both to a .h5 file. 

The code is messy and I'll comment it a bit more later
- Andy
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import pickle
import scipy.signal
import patchify
import glob
import cv2
from skimage.exposure import rescale_intensity
import argparse
from sklearn.cluster import KMeans
from itertools import groupby
from skimage import color, data, restoration
import h5py
import random
from tqdm import tqdm
import os
import time

def specgr (fname,ecen,spec_params,cut_shot=2):
    ece_data = pickle.load(open(fname,'rb'))
    ece_num = '\\tecef%.2i' % (ecen)
    sig_in = ece_data[ece_num][:np.int_(cut_shot*spec_params['fs'])]
    f, t, Sxx = scipy.signal.spectrogram(sig_in, nperseg=spec_params['nperseg'], noverlap=spec_params['noverlap'],fs=spec_params['fs'], window=spec_params['window'],scaling=spec_params['scaling'], detrend=spec_params['detrend'])
    Sxx = np.log(Sxx + spec_params['eps'])
    Sxx=(Sxx-np.min(Sxx))/(np.max(Sxx)-np.min(Sxx))
    Sxx = Sxx[:-1,:];f=f[:-1]
    return Sxx,f,t

def norm(data):
    mn = data.mean()
    std = data.std()
    return((data-mn)/std)

def rescale(data):
    return (data-data.min())/(data.max()-data.min())

def quantfilt(src,thr=0.9):
    filt = np.quantile(src,thr,axis=0)
    out = np.where(src<filt,0,src)
    return out

# gaussian filtering
def gaussblr(src,filt=(31, 3)):
    src = (rescale(src)*255).astype('uint8')
    out = cv2.GaussianBlur(src,filt,0)
    return rescale(out)

# mean filtering
def meansub(src):
    mn = np.mean(src,axis=1)[:,np.newaxis]
    out = np.absolute(src - mn)
    return rescale(out)

# morphological filtering
def morph(src):
    src = (rescale(src)*255).astype('uint8')
    
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    mask = cv2.morphologyEx(src, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        
    return rescale(mask)


if __name__ == '__main__':
    fs = 500000
    spec_params={
        'nperseg': 512, # default 1024
        'noverlap': 256, # default: nperseg / 4
        'fs': fs, # raw signal sample rate is 4MHz
        'window': 'hamm',
        'scaling': 'density', # {'density', 'spectrum'}
        'detrend': 'linear', # {'linear', 'constant', False}
        'eps': 1e-11}

    data_path = "/scratch/gpfs/aonelson/ml_database/ECE_data/"
    flist = glob.glob(data_path+"*.pkl")
    thr = 0.9

    out_file = h5py.File('/scratch/gpfs/ar0535/spectrogram_data.hdf5', 'a')

    for fname in flist:
        shotn = fname[fname.rfind('_')+1:fname.rfind('.')]

        for chn in range(20):
            try:
                s,f,t = specgr(fname,chn+1,spec_params,2)


                # image processing pipeline
                out_quant= quantfilt(s,thr)
                out_gauss=np.empty(s.shape)
                out_mean=np.empty(s.shape)
                out_morph = np.empty(s.shape)
                out_final = np.empty(s.shape)

                out_gauss =  gaussblr(out_quant,(31, 3))
                out_mean = meansub(out_gauss)    
                out_morph = morph(out_mean)
                out_final = meansub(out_morph)

                grp = out_file.create_group('ece_'+shotn+'/chn_'+str(chn+1))
                grp.create_dataset('spec', data=s)
                grp.create_dataset('f', data=f)
                grp.create_dataset('t', data=t)
                grp.create_dataset('pipeline_out', data=out_final)

            except pickle.UnpicklingError as e:
                continue
            except Exception as e:
                print(traceback.format_exc(e))
                continue
    out_file.close()
