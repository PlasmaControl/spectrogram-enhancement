from re import L
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py as h5
import os
import time
import random
import io

from keras import layers
from keras.models import Model, load_model
from keras.callbacks import TensorBoard

import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

JOB_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])

# The database labels
AE_TYPE = np.asarray(['LFM', 'BAE', 'RSAE', 'TAE'])
LABELS = np.asarray(['labels_132710.txt', 'labels_138388.txt',
                     'labels_142111.txt', 'labels_153068.txt',
                     'labels_153592.txt', 'labels_159243.txt', 
                     'labels_159246.txt', 'labels_170660.txt',
                     'labels_170670.txt', 'labels_175987.txt',
                     'labels_176053.txt', 'labels_178631.txt',
                     'labels_178636.txt', 'labels_132240.txt',
                     'labels_163147.txt', 'labels_159242.txt',
                     'labels_159245.txt', 'labels_159247.txt',
                     'labels_178637.txt', 'labels_178640.txt',
                     'labels_170677.txt', 'labels_170679.txt',
                     'labels_178641.txt', 'labels_178872.txt',
                     'labels_170672.txt', 'labels_178633.txt'])

DATA_FILES = np.asarray([ 'ece_132710.pkl', 'ece_138388.pkl',
                          'ece_142111.pkl', 'ece_153068.pkl',
                          'ece_153592.pkl', 'ece_159243.pkl', 
                          'ece_159246.pkl', 'ece_170660.pkl',
                          'ece_170670.pkl', 'ece_175987.pkl',
                          'ece_176053.pkl', 'ece_178631.pkl',
                          'ece_178636.pkl', 'ece_132240.pkl',
                          'ece_163147.pkl', 'ece_159242.pkl',
                          'ece_159245.pkl', 'ece_159247.pkl',
                          'ece_178637.pkl', 'ece_178640.pkl',
                          'ece_170677.pkl', 'ece_170679.pkl',
                          'ece_178641.pkl', 'ece_178872.pkl',
                          'ece_170672.pkl', 'ece_178633.pkl'])

SHOTS = np.asarray([ '132710', '138388',
                          '142111', '153068',
                          '153592', '159243', 
                          '159246', '170660',
                          '170670', '175987',
                          '176053', '178631',
                          '178636', '132240',
                          '163147', '159242',
                          '159245', '159247',
                          '178637', '178640',
                          '170677', '170679',
                          '178641', '178872',
                          '170672', '178633'])

def get_params(JOB_ID):
    '''
    Returns parameters based on job array index
    '''
    if JOB_ID % 2 == 0:
        MULTI = True
    else:
        MULTI = False
        
    if JOB_ID in [0,1,2,3]:
        num_samples = 200
        width = 32
        kernels = [5, 15, 25]
        if JOB_ID in [0,1]:
            nodes = [4, 8, 16]
        else:
            nodes = [2, 4, 8]
    elif JOB_ID in [4,5,6,7]:
        num_samples = 120
        width = 16
        kernels = [5, 11, 15]
        if JOB_ID in [4,5]:
            nodes = [4, 8, 16]
        else:
            nodes = [2, 4, 8]
    elif JOB_ID in [8,9]:
        num_samples = 60
        width = 8
        kernels = [3, 5, 9]
        nodes = [2, 4, 8]
    
    return num_samples, kernels, nodes, width, MULTI

def get_model_name(JOB_ID):
    # Directory holding the models
    dir = '/scratch/gpfs/ar0535/spec_model_data/Multiscale/sweep_models/'
    
    _, kernels, nodes, width, MULTI = get_params(JOB_ID)
    
    # Write out model name based on current job params
    if MULTI:
        label = f'keras_model_{width}_{nodes[2]}_{kernels[0]}_{kernels[1]}_{kernels[2]}'
    else:
        label = f'keras_model_{width}_{nodes[2]}_{kernels[2]}'
    return dir+label

def load_ECE_data(h5_file='/projects/EKOLEMEN/ae_andy/MLP_specs.h5', n_channels=40):
    # Begin loop over a batch of files
    x = []
    y = []
    n_labels = 4 # 4 Modes being detected
    
    with h5.File(h5_file, 'r') as file:
        for k,shot in enumerate(SHOTS):
            Sxx = []
            
            t = np.asarray(file[shot]['ece']['chn_1']['time'])
            f = np.asarray(file[shot]['ece']['chn_1']['freq'])
            for i in range(n_channels):
                s = np.asarray(file[shot]['ece']['chn_'+str(i+1)]['sprectrogram'])
                Sxx.append(s)
            
            M = len(t)
            Sxx = np.dstack(Sxx)

            # Now get the ylabels, only training data has these
            t1, t2, _, _, channels, ylabel = np.loadtxt(
                '/projects/EKOLEMEN/ece_cnn/MLP_baseline/label_directory/' + LABELS[k], skiprows=1,
                unpack=True, delimiter=', ',
                dtype=str
            )
            t1 = np.asarray(t1, float)
            t2 = np.asarray(t2, float)
            channels = np.asarray(channels, int)
            vector_label = np.zeros((M, n_channels, n_labels))
            for i in range(len(channels)):
                if np.any(ylabel[i] == AE_TYPE) and channels[i] - 1 < n_channels:
                    ae_ind = np.ravel(np.where(AE_TYPE == ylabel[i]))[0]
                    t1_interp = np.abs(t1[i] - t).argmin()
                    t2_interp = np.abs(t2[i] - t).argmin()
                    vector_label[t1_interp:t2_interp, channels[i] - 1, ae_ind] = 1.0
                    
            y.append(vector_label)
            x.append(Sxx)  # output data
    return (np.asarray(x), np.asarray(y)), t, f

# patches all the strips together to 1 spectrogram
def patch(arr, window_size, num_strips):
    '''
    Takes an array of full spectrograms and returns an array of 
    time slices of the spectrograms
    
    arr has dimensions: 
    (# full spectrograms, 256 freq, total times: 3905 normally)
    
    all_patches has dimensions:
    (# strips * # full spectrograms, 256 freq, time width of strips) 
    
    Note that some of the spectrogram may be cut if time width * # strips != 3905
    '''
    
    # Unpack sizes and find number of strips
    height = window_size[0]
    width  = window_size[1]
    length = np.shape(arr)[2]
    num_specs = np.shape(arr)[0]

    assert num_strips == np.floor(length / width)
    
    all_patches = np.empty((num_specs * num_strips, height, width))
    for i in range(len(arr)):
        for strip in range(num_strips):
            all_patches[strip + num_strips * i] = arr[i][:,strip*width:(strip+1)*width]
    
    return all_patches

# splits spectrogram into strips
def unpatch(arr, window_size, num_strips):
    '''
    Takes an array of spectrogram slices and returns an array of the full spectrograms.
    
    arr has dimensions: 
    (# strips * # full spectrograms, 256 freq, time width of strips) 
    
    all_spectrograms has dimensions:
    (# full spectrograms, 256 freq, time width of strips * # strips)
    
    Note time dimension may be shrunken if time width * # strips != 3905
    '''
    height = window_size[0]
    width  = window_size[1]
    num_specs = int(len(arr) / num_strips)
    
    assert num_specs * num_strips == len(arr)
    
    all_spectrograms = []
    for spec in range(num_specs):
        reconstructed = np.empty((256, num_strips * width))
        for strip in range(num_strips):
            reconstructed[:,strip*width:(strip+1)*width] = arr[strip + num_strips * spec]

        all_spectrograms.append(reconstructed)
    return np.array(all_spectrograms)

# reshapes the data
def reshape(arr):
    shape = np.shape(arr[0])
    arr = np.reshape(arr, (len(arr), shape[0], shape[1], 1))
    return arr

def process_inputs(x, y, window_size, num_strips):    
    # Keep reshaping and split into test/train
    n_valid = 5
    n_test = 5
    n_train = x.shape[0] - n_valid - n_test
    test_inds = [1, 6, 14, 11, 18]  #, 23] 
    
    valid_inds = [i for i in range(x.shape[0]) if i not in test_inds]
    valid_inds = np.random.choice(valid_inds, n_valid, replace=False)

    # test shots are 138388, 159246, 163147, 178631, 178637, 
    train_inds = [i for i in range(x.shape[0]) if i not in test_inds and i not in valid_inds]
    print(test_inds, valid_inds, train_inds, flush=True)
    
    # Turn spectrograms into patches
    x_test  = patch_spec(x, test_inds, window_size, num_strips)
    x_valid = patch_spec(x, valid_inds, window_size, num_strips)
    x_train = patch_spec(x, train_inds, window_size, num_strips)
    
    # Combine patches in label to single labels
    y_test  = patch_label(y, test_inds, window_size[1], num_strips)
    y_valid = patch_label(y, valid_inds, window_size[1], num_strips)
    y_train = patch_label(y, train_inds, window_size[1], num_strips)
    
    # Double check dimensions are correct
    assert(np.shape(x_test)[0] == np.shape(y_test)[0])
    assert(np.shape(x_train)[0] == np.shape(y_train)[0])
    assert(np.shape(x_valid)[0] == np.shape(y_valid)[0])
    
    return x_test, x_train, x_valid, y_test, y_train, y_valid

def patch_spec(x, inds, window_size, num_strips):
    '''
    Takes x input with dimensions:
    (shots, channels, time, freq)
    
    and turns each spectrogram into num_strips patches of window_size. 
    Output will have dimensions:
    (shots*channels*num_strips, freq, time)
    
    Note on ordering: goes shot by shot, then all strips per channel
    before moving to next channel, then all channels before next shot. 
    This relevant to make sure label patches match spec patches. 
    '''
    channels = np.shape(x)[1]
    output = []
    
    for ind in inds:
        spec = x[ind,:,:,:]
        patches = patch(spec, window_size, num_strips)
        
        if output == []:
            output = patches
        else:
            output = np.append(output, patches, axis=0)
        
    return output
        
def patch_label(y, shots, width, num_strips):
    '''
    Assumes we are only looking at 4 different AE modes
    
    Takes input y with dimensions:
    (shots, channels, time, 4 AE modes)
    
    and turns each label into a single label for each patch. It 
    also looks to see if a mode is active during the entire time
    interval, and sets it active if any time during the width is active. 
    Output will have dimensions:
    (shots*channels*num_strips, 4 AE modes)
    
    Note on ordering: matches patch_spec() order
    '''
    channels = np.shape(x)[1]
    labels = np.zeros((len(shots)*channels*num_strips, 4))
    
    for m, shot in enumerate(shots):
        for chn in range(channels):
            for strip in range(num_strips):
                ind = m*len(shots) + chn * channels + strip
                t_ind = strip*width
                labels[ind,:] = np.any(y[shot,chn,t_ind:t_ind+width,:], axis=0)
    
    return labels

if __name__ == '__main__':
    start = time.time()
    
    num_samples, kernels, nodes, width, MULTI = get_params(JOB_ID)
    window_size = (256, width)
    label_window = (4, width)
    
    # 1. Load Alan's labeled training data + process into appropriate slices (maybe allow overlapping if I need more slices, but atm unsure)
    x, y, t, f = load_ECE_data()
    num_strips = int(np.floor(t / window_size[1]))
    
    # Split into test, train, valid and form into patches
    x_test, x_train, x_valid, y_test, y_train, y_valid = process_inputs(x, y, window_size, num_strips)
    
    # 2. Grab models from saved folder
    autoencoder = load_model(get_model_name(JOB_ID))
    
    # 3. Split model into full model and just encoder
    if MULTI:
        last_encoder = 21 # Need to double check this
    else:
        last_encoder = 6  # Need to double check this
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[last_encoder].output)
    
    
    # 4. Train basic MLP of similar form to Alan's, for both latent and decoded data sets
    #       Keep layer size and num of layers consistent for time widths, but can use smaller layers for smaller widths
    
    
    print(f'Total time: {int((time.time()-start)/60)} min')