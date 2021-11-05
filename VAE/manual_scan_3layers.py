import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy.signal
import glob
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
from patchify import patchify, unpatchify
import glob
# import tensorflow

from keras import layers
from keras.datasets import mnist
from keras.models import Model
from keras.models import load_model
from keras.callbacks import EarlyStopping

# patches all the strips together to 1 spectrogram
def patch(arr):
    all_patches = np.empty((len(arr)* 30, 256, 128))
    for i in range(len(arr)):
        patches = patchify(arr[i], (256, 128), step=128)
        
        for x in range(30):
            all_patches[(x + 30 * i)] = patches[0][x]
    
    return all_patches

# splits spectrogram into strips
def unpatch(arr):
    all_spectrograms = []
    for i in range(int(len(arr) / 30)):
        Sxx = []
        for x in range(30):
            Sxx.append(arr[x + 30 * i])
        
        y=[Sxx]
        reconstructed = unpatchify(np.array(y), (256, 3840))
        all_spectrograms.append(reconstructed)
    return np.array(all_spectrograms)

#reshapes the data
def reshape(arr):
    arr = np.reshape(arr, (len(arr), 256, 128, 1))
    return arr

# displays Sxx and final
def display(Sxx, final, fname, dset):
    n = 5
    
    t = np.array(dset['t'])[:3840]
    f = (np.array(dset['f'])/1000)+1
    
    idx = np.random.randint(len(Sxx), size=n)
    plots1 = Sxx[idx, :]
    plots2 = final[idx, :]
    
    fig = plt.figure(figsize=(8,12))
    grd = gridspec.GridSpec(ncols=1, nrows=(2 * n), figure=fig)
    ax=[None] * (2 * n)
    
    for i, (plot1, plot2) in enumerate(zip(plots1, plots2)):
        ax[2*i] = fig.add_subplot(grd[2*i])
        ax[2*i].pcolormesh(t,f,plot1,cmap='hot',shading='gouraud')
        _=plt.ylabel('Original (kHz)')
        
        ax[2*i+1] = fig.add_subplot(grd[2*i+1])
        ax[2*i+1].pcolormesh(t,f,plot2,cmap='hot',shading='gouraud')
        _=plt.ylabel('Final (kHz)')
    
    plt.savefig(fname)
    
def plt_spec_shot(dset, predictions, noisy, shotn, i, plot_name):
    # Read data from hdf5 file and change shape of raw spectrogram
    pipeline = []
    pipeline.append(np.array(dset['pipeline_out']))

    # Change shape of predictions and processed data for viewing
    predictions = np.squeeze(predictions, axis=3)
    noisy = unpatch(noisy)[0,:,:]
    predictions = unpatch(predictions)[0,:,:]
    processed = unpatch(patch(pipeline))[0,:,:]
    
    t = np.array(dset['t'])[:3840]
    f = (np.array(dset['f'])/1000)+1
    
    # Make plot 
    fig = plt.figure(figsize=(8,12))
    grd = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
    ax=[None] * 3
    caption = 'shot# %s, channel %i' % (shotn, i)
    
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
    
    plt.savefig(plot_name)


if __name__ == '__main__':
    # Sweeping values
    ker_vals = [(5,5)]
    conv1_vals = [16]
    conv2_vals = [32]
    conv3_vals = [64]
    
    # Smaller params for testing
    # ker_vals = [(3,3)]
    # conv1_vals = [16, 32]
    # conv2_vals = [16]
    # conv3_vals = [16]
    nodes = 32
    
    spectrograms = []
    final = []

    file = h5py.File('/scratch/gpfs/ar0535/spectrogram_data.hdf5', 'r')

    num_samples = 200
    random_sample = random.sample(file.keys(), num_samples)

    for fname in tqdm(random_sample):
        shotn = fname[fname.rfind('_')+1:fname.rfind('.')]

        for chn in range(20):
            name = fname+'/chn_'+str(chn+1)
            spectrograms.append(np.array(file[name]['spec']))     
            final.append(np.array(file[name]['pipeline_out']))
    
    # Change shape so that time length is 128 points
    spectrograms = patch(spectrograms)
    final = patch(final)

    #split into 60% (train), 25% (tune), 15% (test)
    Sxx_train, Sxx_tune, Sxx_test = np.split(spectrograms, [int(len(spectrograms)*0.6), int(len(spectrograms)*0.85)])
    final_train, final_tune, final_test = np.split(final, [int(len(final)*0.6), int(len(final)*0.85)])
    
    # reshape our data to add 1 extra dim for pooling later
    Sxx_train_reshaped = reshape(Sxx_train)
    Sxx_test_reshaped = reshape(Sxx_test)
    Sxx_tune_reshaped = reshape(Sxx_tune)
    final_train_reshaped = reshape(final_train)
    final_test_reshaped = reshape(final_test)
    final_tune_reshaped = reshape(final_tune)
    
    # Initialize optimal params
    best_val_loss = 1
    best_model = 0
    best_ind = (-1, -1, -1, -1)
    val_losses = np.zeros((len(ker_vals), len(conv1_vals), len(conv2_vals), len(conv3_vals)))
    best_hist = np.array([0])
    pred_times = np.zeros((len(ker_vals), len(conv1_vals), len(conv2_vals), len(conv3_vals)))
    
    # Stops the model early
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    ep = 100 # Epochs
    
    # Time total training so I can estimate total run times
    model_start_time = time.time()
    
    # Counter for printing so I can see where the code is
    count = 0
    for ker_in, ker_val in enumerate(ker_vals):
        for conv1_in, conv1_val in enumerate(conv1_vals):
            for conv2_in, conv2_val in enumerate(conv2_vals):
                for conv3_in, conv3_val in enumerate(conv3_vals):
                    # Initialize network
                    input = layers.Input(shape = (256, 128, 1))

                    x = layers.Conv2D(conv1_val, ker_val, activation="relu", padding="same")(input)
                    x = layers.MaxPooling2D((2, 2), padding="same")(x)
                    x = layers.Conv2D(conv2_val, ker_val, activation="relu", padding="same")(x)
                    x = layers.MaxPooling2D((2, 2), padding="same")(x)
                    x = layers.Conv2D(conv3_val, ker_val, activation="relu", padding="same")(x)
                    x = layers.MaxPooling2D((2, 2), padding="same")(x)

                    x = layers.Conv2DTranspose(conv3_val, ker_val, strides=2, activation="relu", padding="same")(x)
                    x = layers.Conv2DTranspose(conv2_val, ker_val, strides=2, activation="relu", padding="same")(x)
                    x = layers.Conv2DTranspose(conv1_val, ker_val, strides=2, activation="relu", padding="same")(x)
                    x = layers.Conv2D(1, ker_val, activation="sigmoid", padding="same")(x)

                    autoencoder = Model(input, x)
                    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
                        
                    hist = autoencoder.fit(
                        x=Sxx_train_reshaped,
                        y=final_train_reshaped,
                        epochs=ep,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(Sxx_tune_reshaped, final_tune_reshaped),
                        verbose=1,
                        # callbacks=[es],
                    )
                        
                    # Save validation loss to compare
                    val_loss = hist.history['val_loss'][-1]
                    val_losses[ker_in, conv1_in, conv2_in, conv3_in] = val_loss
                        
                    # See if best validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = autoencoder
                        best_ind = (ker_in, conv1_in, conv2_in, conv3_in)
                        best_hist = hist
                        
                    # Test how long it takes to make predictions
                    t_predict = 0.0
                    chn_num = 20 # Total channel number
                    for i in range(chn_num):
                        # Load specific channel data
                        dset = file[f'ece_{176053}'][f'chn_{i+1}']

                        # Read raw data from hdf5 file and change shape of raw spectrogram
                        noisy = []
                        noisy.append(np.array(dset['spec']))
                        noisy = patch(noisy)

                        # Time autoencoder predictions
                        start = time.time()
                        predictions = autoencoder.predict(reshape(noisy))
                        end = time.time()

                        # Add prediction time to running total
                        t_predict += (end-start) / np.shape(noisy)[0]

                    # Time to get prediction
                    t_predict /= chn_num
                    pred_times[ker_in, conv1_in, conv2_in, conv3_in] = t_predict / chn_num
                        
                    # Print current count so I can see progress
                    count += 1
                    print(count)
                        
    # Print summary of best model
    best_model.summary()
    
    # Make directory to save model
    data_path = f'/scratch/gpfs/ar0535/spec_model_data/sweep_3layer/'
    os.makedirs(data_path)
    
    # Save autoencoder Model
    best_model.save(data_path+'best_model')    
    
    # predict and reformat
    predictions = best_model.predict(Sxx_test_reshaped)
    predictions = np.squeeze(predictions, axis=3)
    
    # restitch everything together to a list of spectrograms
    noisy = unpatch(Sxx_test)
    autoencoder_final = unpatch(predictions)
    pipeline_final = unpatch(final_test)
    
    # Sample data set for general time and freq data (axis for plotting)
    shotn = 176053 # Shot we decide to look at
    dset = file[f'ece_{shotn}']['chn_1']
    display(noisy, autoencoder_final, data_path+'ex_specs.png', dset)
    
    plt.clf()
    # Save validation loss and validation loss plot
    y = best_hist.history['val_loss']
    plt.plot(range(len(y)), y)
    plt.savefig(data_path+'best_val_loss.png')
    np.save(data_path+'val_losses.npy', val_losses)
    
    # Get a few spectrogram plots w/ best model
    for i in range(10,13):
        # Load specific channel data
        dset = file[f'ece_{shotn}'][f'chn_{i+1}']

        # Read raw data from hdf5 file and change shape of raw spectrogram
        noisy = []
        noisy.append(np.array(dset['spec']))
        noisy = patch(noisy)

         # Make predictions
        predictions = best_model.predict(reshape(noisy))

        # Plot raw, processed, and predicted spectrograms
        plt_spec_shot(dset, predictions, noisy, shotn, i+1, data_path+f'plot_chn_{i+1}.png')
    
    # Compare validation losses and prediction times for parameters averaged over all other params
    ker_avg_loss = np.zeros((len(ker_vals),1))
    conv1_avg_loss = np.zeros((len(conv1_vals),1))
    conv2_avg_loss = np.zeros((len(conv2_vals),1))
    conv3_avg_loss = np.zeros((len(conv3_vals),1))
    ker_pred_avg = np.zeros((len(ker_vals),1))
    conv1_pred_avg = np.zeros((len(conv1_vals),1))
    conv2_pred_avg = np.zeros((len(conv2_vals),1))
    conv3_pred_avg = np.zeros((len(conv3_vals),1))
    
    # Kernel
    for ker_in in range(len(ker_vals)):
        # Val loss
        flat = val_losses[ker_in,:,:,:].flatten()
        ker_avg_loss[ker_in] = np.average(flat)
        
        # Pred time
        flat = pred_times[ker_in,:,:,:].flatten()
        ker_pred_avg[ker_in] = np.average(flat)
        
    # Convolutional Layer 1 
    for conv1_in in range(len(conv1_vals)):
        flat = val_losses[:,conv1_in,:,:].flatten()
        conv1_avg_loss[conv1_in] = np.average(flat)
        
        # Pred time
        flat = pred_times[:,conv1_in,:,:].flatten()
        conv1_pred_avg[conv1_in] = np.average(flat)
        
    # Convolutional Layer 2 
    for conv2_in in range(len(conv2_vals)):
        flat = val_losses[:,:,conv2_in,:].flatten()
        conv2_avg_loss[conv2_in] = np.average(flat)
        
        # Pred time
        flat = pred_times[:,:,conv2_in,:].flatten()
        conv2_pred_avg[conv2_in] = np.average(flat)
    
    # Convolutional Layer 3
    for conv3_in in range(len(conv3_vals)):
        flat = val_losses[:,:,:,conv3_in].flatten()
        conv3_avg_loss[conv3_in] = np.average(flat)
        
        # Pred time
        flat = pred_times[:,:,:,conv3_in].flatten()
        conv3_pred_avg[conv3_in] = np.average(flat)
        
    # Save arrays for comparing losses by parameter
    np.savez(data_path+'loss_comparisons.npz', ker_loss=ker_avg_loss, conv1_loss=conv1_avg_loss, 
             conv2_loss=conv2_avg_loss, conv3_loss=conv3_avg_loss, ker_time=ker_pred_avg, 
             conv1_time=conv1_pred_avg, conv2_time=conv2_pred_avg, conv3_time=conv3_pred_avg)
    
    total_time = time.time() - model_start_time
    print(total_time)
    
    file.close()
