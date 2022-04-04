import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py as h5
from co2_deps import *
from matplotlib import gridspec

def load_time_series_tensor(shot, chord1, chord2, file):
    # `fid` indicates file ID found in the time series folder
    # `chord1` must be `v1` or `v2`
    # `chord2` must be `v3` or `r0`
    
    if chord1=='v1' and chord2=='v3':
        signal1 = file[str(shot)]['co2']['dp1v1uf']
        signal2 = file[str(shot)]['co2']['dp1v3uf']
    if chord1=='v2' and chord2=='r0':
        signal1 = file[str(shot)]['co2']['dp1v2uf']
        signal2 = file[str(shot)]['co2']['dp1r0uf']
    
    time = file[str(shot)]['co2']['co2_time']

    return signal1, signal2, time

def check_magnetics_data(file, shot):
    '''
    Makes sure all the needed magnetics data is in the h5 file
    '''
    check1 = 'dp1v1uf'  in file[shot]['co2'].keys()
    check2 = 'dp1v3uf'  in file[shot]['co2'].keys()
    check3 = 'dp1v2uf'  in file[shot]['co2'].keys()
    check4 = 'dp1r0uf'  in file[shot]['co2'].keys()
    check5 = 'co2_time' in file[shot]['co2'].keys()
    
    return (check1 and check2 and check3 and check4 and check5)


if __name__ == '__main__':
    '''
    This file goes through all the shots in the big hdf5 file that have CO2 data. 
    It will take the crosspower spectrograms for all of them and then save the data to 
    the hdf5 file that the data came from. 
    '''
    
    with h5.File('/projects/EKOLEMEN/ae_andy/AE_data.h5','r+') as file:
        shots = file.keys()

        for i, shot in enumerate(shots):
            # Double check the need folders and data files exist
            if 'co2' in file[shot].keys():
                if check_magnetics_data(file, shot):
                    # --- Get Magnetics signals ---
                    signal1, _, _ = load_time_series_tensor(shot, 'v1', 'v3', file)
                    _, signal2, t = load_time_series_tensor(shot, 'v2', 'r0', file)

                    # --- Calculate Spectrogram ---
                    ampsp, freq, time = ae_co2(np.array(signal1), np.array(signal2), np.array(t))

                    # --- Save to hdf5 file ---
                    file[shot]['co2'].create_dataset('spectrogram', data=ampsp)
                    file[shot]['co2'].create_dataset('spec_freq', data=freq)
                    file[shot]['co2'].create_dataset('spec_time', data=time)

            if i % 10 == 0:
                print('Percent done: {}%'.format(round(i*100.0/len(shots))), flush=True)

    print('Done!', flush=True)
