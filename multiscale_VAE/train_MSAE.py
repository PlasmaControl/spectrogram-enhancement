import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py as h5
import os
import time
import random
import io

from keras import layers
from keras.models import Model
import scipy as sp
from tensorflow import keras

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

JOB_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])

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

# displays Sxx and final
def display(noisy, processed, predictions, datapath, dset, n, window_size, num_strips):
    tmax = num_strips * window_size[1]
    t = np.array(dset['time'])[:tmax]
    f = (np.array(dset['freq'])/1000)+1
    
    for i in range(n):
        # Make plot 
        fig = plt.figure(figsize=(8,12))
        grd = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
        ax=[None] * 3

        ax[0] = fig.add_subplot(grd[0])
        ax[0].pcolormesh(t,f,noisy[i,:,:],cmap='hot',shading='gouraud')
        _=plt.ylabel('Original - Raw Data (kHz))')

        ax[1] = fig.add_subplot(grd[1])
        ax[1].pcolormesh(t,f,predictions[i,:,:],cmap='hot',shading='gouraud')
        _=plt.ylabel('Predicted Denoised (kHz)')
        
        ax[2] = fig.add_subplot(grd[2])
        ax[2].pcolormesh(t,f,processed[i,:,:],cmap='hot',shading='gouraud')
        _=plt.ylabel('Pipeline (kHz)')

        fname = datapath+'ex_spec'+str(i)+'.png'
        plt.savefig(fname)
        # plt.show()

# Plots a spectrogram shot
def plt_spec_shot(dset, predictions, noisy, caption, plot_name, window_size, num_strips):
    # Read data from hdf5 file and change shape of raw spectrogram
    pipeline = []
    pipeline.append(np.array(dset['pipeline_out']))

    # Change shape of predictions and processed data for viewing
    predictions = np.squeeze(predictions, axis=3)
    noisy = unpatch(noisy, window_size, num_strips)[0,:,:]
    predictions = unpatch(predictions, window_size, num_strips)[0,:,:]
    processed = unpatch(patch(pipeline, window_size, num_strips), window_size, num_strips)[0,:,:]
    
    tmax = num_strips*window_size[1]
    t = np.array(dset['time'])[:tmax]
    f = (np.array(dset['freq'])/1000)+1
    
    # Make plot 
    fig = plt.figure(figsize=(8,12))
    grd = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
    ax=[None] * 3
    
    ax[0] = fig.add_subplot(grd[0])
    ax[0].pcolormesh(t,f,noisy,cmap='hot',shading='gouraud')
    _=plt.ylabel('Original - Raw Data (kHz))')
    ax[0].set(title=caption)

    ax[1] = fig.add_subplot(grd[1])
    ax[1].pcolormesh(t,f,predictions,cmap='hot',shading='gouraud')
    _=plt.ylabel('Predicted Denoised (kHz)')
    
    ax[2] = fig.add_subplot(grd[2])
    ax[2].pcolormesh(t,f,processed,cmap='hot',shading='gouraud')
    _=plt.ylabel('Pipeline (kHz)')
    
    # plt.show()
    # plt.savefig(plot_name)
    
    return fig
    
def MSConv2D(initial, nodes, kernels):
    '''
    Creates layers for Multi-scale 2D Convolution w/ 3 different kernel sizes
    '''
    # Kernel 1
    x1 = layers.Conv2D(nodes, (kernels[0],kernels[0]), activation="relu", padding="same")(initial)
    x1 = layers.MaxPooling2D((2, 2), padding="same")(x1)
    
    # Kernel 2
    x2 = layers.Conv2D(nodes, (kernels[1],kernels[1]), activation="relu", padding="same")(initial)
    x2 = layers.MaxPooling2D((2, 2), padding="same")(x2)
    
    # Kernel 3
    x3 = layers.Conv2D(nodes, (kernels[2],kernels[2]), activation="relu", padding="same")(initial)
    x3 = layers.MaxPooling2D((2, 2), padding="same")(x3)
    
    # Sum together
    return layers.Add()([x1, x2, x3])

def MSConv2DTranspose(initial, nodes, kernels):
    '''
    Creates layers for Multi-scale 2D Deconvolution w/ 3 different kernel sizes 
    '''
    # Kernel 1
    x1 = layers.Conv2DTranspose(nodes, (kernels[0],kernels[0]), strides=2, activation="relu", padding="same")(initial)
    
    # Kernel 2
    x2 = layers.Conv2DTranspose(nodes, (kernels[1],kernels[1]), strides=2, activation="relu", padding="same")(initial)
    
    # Kernel 3
    x3 = layers.Conv2DTranspose(nodes, (kernels[2],kernels[2]), strides=2, activation="relu", padding="same")(initial)
    
    # Sum together
    return layers.Add()([x1, x2, x3])

# Get random samples to train model
def get_samples(file, num_samples, window_size, num_strips, Split=True):
    spectrograms = []
    final = []
    
    random_sample = random.sample(file.keys(), num_samples)

    for shotn in random_sample:
        for chn in range(20):
            if f'chn_{chn+1}' in file[shotn]['ece'].keys():
                spectrograms.append(np.array(file[shotn]['ece'][f'chn_{chn+1}']['spectrogram']))     
                final.append(np.array(file[shotn]['ece'][f'chn_{chn+1}']['pipeline_out']))
    
    # Change shape so that time length is 128 points
    spectrograms = patch(spectrograms, window_size, num_strips)
    final = patch(final, window_size, num_strips)

    if Split:
        # Shuffle slices to remove errors from different shots
        seed = 239517
        np.random.RandomState(seed)
        np.random.shuffle(spectrograms)
        np.random.RandomState(seed)
        np.random.shuffle(final)
                
        ### Returns spectrograms split into training, testing, and validation
        #split into 60% (train), 25% (tune), 15% (test)
        Sxx_train, Sxx_tune, Sxx_test = np.split(spectrograms, [int(len(spectrograms)*0.6), int(len(spectrograms)*0.85)])
        final_train, final_tune, final_test = np.split(final, [int(len(final)*0.6), int(len(final)*0.85)])
        
        # reshape our data to add 1 extra dim for pooling later
        Sxx_train_reshaped = reshape(Sxx_train)
        Sxx_test_reshaped = reshape(Sxx_test)
        Sxx_tune_reshaped = reshape(Sxx_tune)
        final_train_reshaped = reshape(final_train)
        # final_test_reshaped = reshape(final_test)
        final_tune_reshaped = reshape(final_tune)
        
        return (Sxx_train_reshaped, Sxx_test_reshaped, Sxx_tune_reshaped, final_train_reshaped, final_test, final_tune_reshaped, Sxx_test)
    else: 
        ### Return spectrograms in one group
        return (reshape(spectrograms), final)

# saves plots and losses
def post_process(file, autoencoder, hist, kernels, n, window_size, num_strips):
    '''
    Plot predictions for spectrograms and losses
    '''
    # Make directory to save model
    data_path = f'/scratch/gpfs/ar0535/spec_model_data/Multiscale/t_width_{window_size[1]}/Ker_{kernels[0]}{kernels[1]}{kernels[2]}/'
    os.makedirs(data_path)
    
    # Save autoencoder Model
    autoencoder.save(data_path+'keras_model')
    
    '''
    raw_specs, pipeline_specs = get_samples(file, n, window_size, num_strips, Split=False)
    
    ### Pick random data to plot (Done here bc I ran into memory errors)
    # predict and reformat
    predictions = autoencoder.predict(raw_specs)
    predictions = np.squeeze(predictions, axis=3)
    
    # restitch everything together to a list of spectrograms
    raw_specs_reshaped = np.squeeze(raw_specs, axis=3)
    noisy = unpatch(raw_specs_reshaped, window_size, num_strips)
    autoencoder_final = unpatch(predictions, window_size, num_strips)
    pipeline_specs = unpatch(pipeline_specs, window_size, num_strips)
    
    # Sample data set for general time and freq data (axis for plotting)
    shotn = '176053' # Shot we decide to look at
    dset = file[shotn]['ece']['chn_1']
    display(noisy, pipeline_specs, autoencoder_final, data_path, dset, n, window_size, num_strips)
    
    
    plt.clf()
    # Save validation loss and validation loss plot
    val_loss = hist.history['val_loss']
    train_loss = hist.history['loss']
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    plt.savefig(data_path+'val_loss.png')
    '''

    shotn = '176053' # Shot we decide to look at
    dset = file[shotn]['ece']['chn_1']
    # Example prediction plot
    logdir = f"/scratch/gpfs/ar0535/logs/Multiscale/logs/{JOB_ID}/plots/"
    file_writer = tf.summary.create_file_writer(logdir)
    for i in range(10, 13):
        # Load specific channel data
        dset = file[shotn]['ece'][f'chn_{i+1}']
        
        # Read raw data from hdf5 file and change shape of raw spectrogram
        noisy = []
        noisy.append(np.array(dset['spectrogram']))
        noisy = patch(noisy, window_size, num_strips)
        
        # Predict spectrograms
        predictions = autoencoder.predict(reshape(noisy))
        
        # Plot raw, processed, and predicted spectrograms
        caption = caption = 'shot# '+ shotn +f', channel {i+1}'
        fig = plt_spec_shot(dset, predictions, noisy, caption, data_path+f'plot_chn_{i+1}.png', window_size, num_strips)
        
        # Save plot in log
        with file_writer.as_default():
            tf.summary.image(f"chn_{i+1}", plot_to_image(fig), step=0)
            
    
def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def get_ker(KER_IDX):
    if KER_IDX == 0:
        return [3, 5, 7]
    elif KER_IDX == 1:
        return [5, 9, 11]
    elif KER_IDX == 2:
        return [7, 11, 15]
    elif KER_IDX == 3:
        return [7, 15, 25]
    else:
        raise(ValueError('KER_IDX invalid'))

WIDTH_VALS = [4, 8, 16, 32]
KER_VALS = [7, 11, 15, 25]
NODE_VALS  = [2, 4, 8]

# TEST
WIDTH_VALS = [4, 8]
KER_VALS = [7, 11]
NODE_VALS  = [2, 4]

# Get from job array ID
WIDTH_IDX = JOB_ID % 4
KER_IDX   = int(JOB_ID / 4) % 4
NODE_IDX  = int(JOB_ID / 16) % 3

# Get from job array ID TEST
WIDTH_IDX = JOB_ID % 2
KER_IDX   = int(JOB_ID / 2) % 2
NODE_IDX  = int(JOB_ID / 4) % 2

HP_T_WIDTH  = hp.HParam('t_width',  hp.Discrete(WIDTH_VALS))
HP_KER_MAX  = hp.HParam('max_ker',  hp.Discrete(KER_VALS))
HP_NODE_MIN = hp.HParam('max_node', hp.Discrete(NODE_VALS))

if __name__ == '__main__':
    start = time.time()
    
    # Samples (will be 20*num_samples because 20 channels)
    # Also scale number of samples so that they all have similar total number
    num_samples = int(128 * 4 / WIDTH_VALS[WIDTH_IDX])

    # Multiscale w/ 5x5, 15x15, and 25x25 kernels
    # kernels = [5, 15, 25]
    # nodes = [8, 16, 32]
    
    kernels = get_ker(KER_IDX)
    nodes = NODE_VALS[NODE_IDX] * np.array([1, 2, 4]) 
    ep = 5 # Epochs, 10 may be too few but 100 was overkill

    window_size = (256, WIDTH_VALS[WIDTH_IDX])
    num_strips = int(np.floor(3905 / window_size[1]))

    with h5.File('/projects/EKOLEMEN/ae_andy/AE_data.h5', 'r') as file:
        # Get data (uses normalizer model to normalize ECE data)
        (Sxx_train_reshaped, _, Sxx_tune_reshaped, final_train_reshaped, 
         _, final_tune_reshaped, _) = get_samples(file, num_samples, window_size, num_strips)

        # Initialize network
        input = layers.Input(shape = (window_size[0], window_size[1], 1))

        x = MSConv2D(input, nodes[0], kernels)
        x = MSConv2D(x, nodes[1], kernels)
        x = MSConv2D(x, nodes[2], kernels)

        x = MSConv2DTranspose(x, nodes[2], kernels)
        x = MSConv2DTranspose(x, nodes[1], kernels)
        x = MSConv2DTranspose(x, nodes[0], kernels)

        # End with normal 3x3 Convolutional Layer
        x = layers.Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

        autoencoder = Model(input, x)
        autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
        autoencoder.summary()

        logdir = f"/scratch/gpfs/ar0535/logs/Multiscale/{JOB_ID}/"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir=logdir, histogram_freq=1)
        
        hparams = {
          HP_T_WIDTH: WIDTH_VALS[WIDTH_IDX],
          HP_KER_MAX: KER_VALS[KER_IDX],
          HP_NODE_MIN: NODE_VALS[NODE_IDX],
        }
        
        with tf.summary.create_file_writer(logdir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            hist = autoencoder.fit(
                x=Sxx_train_reshaped,
                y=final_train_reshaped,
                epochs=ep,
                batch_size=32,
                shuffle=True,
                validation_data=(Sxx_tune_reshaped, final_tune_reshaped),
                verbose=2,
                callbacks=[tensorboard_callback,hp.KerasCallback(logdir, hparams)],
            )
        
        ### Make some plots and save errors
        n = 5 # Number of random test data spectrograms to plot
        # post_process(file, autoencoder, hist, kernels, n, window_size, num_strips)
        
        mins = int(np.floor((time.time() - start) / 60.0))
        print(f'Total time: {mins} min', flush=True)