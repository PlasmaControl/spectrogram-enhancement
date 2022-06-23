import time
import numpy as np
import random
import h5py as h5

from keras.models import load_model

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

# reshapes the data
def reshape(arr):
    shape = np.shape(arr[0])
    arr = np.reshape(arr, (len(arr), shape[0], shape[1], 1))
    return arr

def get_params(JOB_ID):
    '''
    Returns parameters based on job array index
    '''
    if JOB_ID % 2 == 0:
        MULTI = True
    else:
        MULTI = False
        
    if JOB_ID in [0,1,2,3]:
        num_samples = 20
        width = 32
        kernels = [5, 15, 25]
        if JOB_ID in [0,1]:
            nodes = [4, 8, 16]
        else:
            nodes = [2, 4, 8]
    elif JOB_ID in [4,5,6,7]:
        num_samples = 12
        width = 16
        kernels = [5, 11, 15]
        if JOB_ID in [4,5]:
            nodes = [4, 8, 16]
        else:
            nodes = [2, 4, 8]
    elif JOB_ID in [8,9]:
        num_samples = 6
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

# Get random samples to train model
def get_samples(file, num_samples, window_size, num_strips):
    spectrograms = []
    
    random_sample = random.sample(file.keys(), num_samples)
    for shotn in random_sample:
        for chn in range(20):
            if f'chn_{chn+1}' in file[shotn]['ece'].keys():
                spectrograms.append(np.array(file[shotn]['ece'][f'chn_{chn+1}']['spectrogram']))
    
    # Change shape so that time length is window_size[1] points
    spectrograms = patch(spectrograms, window_size, num_strips)

    return reshape(spectrograms)

if __name__ == '__main__':
    '''
    Run autoencoder models to get time estimates on prediction
    '''
    
    channels = 20
    for JOB_ID in range(10):
        num_samples, kernels, nodes, width, MULTI = get_params(JOB_ID)
        window_size = (256, width)
        num_strips = int(np.floor(3905 / window_size[1]))
        
        with h5.File('/projects/EKOLEMEN/ae_andy/AE_data.h5', 'r') as file: 
            samples = get_samples(file, num_samples, window_size, num_strips)
        
        model = load_model(get_model_name(JOB_ID))
        
        start = time.time()
        model.predict(samples)
        avg = (time.time()-start)/(num_samples*channels*num_strips)
        
        print('', flush=True)
        print(f'Width: {width}', flush=True)
        print(f'Multi: {MULTI}', flush=True)
        print(f'Nodes: {nodes}', flush=True)
        if MULTI:
            print(f'Kernels: {kernels[0]} {kernels[1]} {kernels[2]}', flush=True)
        else:
            print(f'Kernel: {kernels[2]}', flush=True)
        print(f'Average pred time (ms): {avg*1000}', flush=True)
        
    