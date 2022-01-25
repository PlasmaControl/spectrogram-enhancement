import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py
import random
import os
from patchify import patchify, unpatchify

from keras import layers
from keras.models import Model

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

# reshapes the data
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

# Plots a spectrogram shot
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

# saves plots and losses
def post_process(file, autoencoder, hist):
    '''
    Plot predictions for spectrograms and losses
    
    Sxx_test_reshaped
    Sxx_test
    final_test
    '''
    # Make directory to save model
    data_path = f'/scratch/gpfs/ar0535/spec_model_data/Multiscale/Ker_'+str(kernels[0])+str(kernels(1))+str(kernels(2))
    os.makedirs(data_path)
    
    # Save autoencoder Model
    autoencoder.save(data_path+'keras_model')    
    
    # predict and reformat
    predictions = autoencoder.predict(Sxx_test_reshaped)
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
    val_loss = hist.history['val_loss']
    plt.plot(range(ep), val_loss)
    plt.savefig(data_path+'val_loss.png')
    np.savetxt(data_path+'val_loss.txt', val_loss)


    chn_num = 20 # Total channel number
    # Example prediction plot
    for i in range(chn_num):
        # Load specific channel data
        dset = file[f'ece_{shotn}'][f'chn_{i+1}']
        
        # Read raw data from hdf5 file and change shape of raw spectrogram
        noisy = []
        noisy.append(np.array(dset['spec']))
        noisy = patch(noisy)
        
        # Predict spectrograms
        predictions = autoencoder.predict(reshape(noisy))
        
        # Plot raw, processed, and predicted spectrograms
        if (i+1) in range(10,13):
            plt_spec_shot(dset, predictions, noisy, shotn, i+1, data_path+f'plot_chn_{i+1}.png')

# Get random samples to train model
def get_samples(file, num_samples, spectrograms, final):
    random_sample = random.sample(file.keys(), num_samples)

    for fname in random_sample:
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
    # final_test_reshaped = reshape(final_test)
    final_tune_reshaped = reshape(final_tune)
    
    return (Sxx_train_reshaped, Sxx_test_reshaped, Sxx_tune_reshaped, final_train_reshaped, final_test, final_tune_reshaped, Sxx_test)
    
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

if __name__ == '__main__':
    # Samples (will be 20*num_samples because 20 channels)
    num_samples = 20
    
    # Multiscale w/ 1x1, 3x3, and 5x5 kernels
    kernels = [1, 3, 5]
    nodes = 32
    
    spectrograms = []
    final = []

    file = h5py.File('/scratch/gpfs/ar0535/spectrogram_data.hdf5', 'r')
    
    # Get data
    (Sxx_train_reshaped, Sxx_test_reshaped, Sxx_tune_reshaped, \
     final_train_reshaped, final_test, final_tune_reshaped, Sxx_test) = \
         get_samples(file, num_samples, spectrograms, final)
    
    # Initialize network
    input = layers.Input(shape = (256, 128, 1))
    
    x = MSConv2D(input, nodes, kernels)
    x = MSConv2D(x, nodes, kernels)
    x = MSConv2D(x, nodes, kernels)
    
    x = MSConv2DTranspose(x, nodes, kernels)
    x = MSConv2DTranspose(x, nodes, kernels)
    x = MSConv2DTranspose(x, nodes, kernels)
    
    # End with normal 3x3 Convolutional Layer
    x = layers.Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(input, x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.summary()

    ep = 10 # Epochs, 10 may be too few but 100 was overkill
    hist = autoencoder.fit(
        x=Sxx_train_reshaped,
        y=final_train_reshaped,
        epochs=ep,
        batch_size=128,
        shuffle=True,
        validation_data=(Sxx_tune_reshaped, final_tune_reshaped),
    )
    
    # Make some plots and save errors
    post_process(file, autoencoder, hist,Sxx_test_reshaped, Sxx_test, final_test)
    
    # Close h5 data file
    file.close()