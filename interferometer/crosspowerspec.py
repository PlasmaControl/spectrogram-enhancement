import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py as h5
from co2_deps import *
from matplotlib import gridspec

def load_time_series_tensor(fid, chord1, chord2):
    #+ `fid` indicates file ID found in the time series folder
    #+ `chord1` must be `v1` or `v2`
    #+ `chord2` must be `v3` or `r0`

    if chord1=='v1' and chord2=='v3':
        signal1 = h5.File(f'/projects/EKOLEMEN/agarcia/time_series/v1v3/signal1_{fid}.h5', 'r')['dp1v1uf'][()]
        signal2 = h5.File(f'/projects/EKOLEMEN/agarcia/time_series/v1v3/signal2_{fid}.h5', 'r')['dp1v3uf'][()]
    if chord1=='v2' and chord2=='r0':
        signal1 = h5.File(f'/projects/EKOLEMEN/agarcia/time_series/v2r0/signal1_{fid}.h5', 'r')['dp1v2uf'][()]
        signal2 = h5.File(f'/projects/EKOLEMEN/agarcia/time_series/v2r0/signal2_{fid}.h5', 'r')['dp1r0uf'][()]

    shots = h5.File(f'/projects/EKOLEMEN/agarcia/time_series/shots_{fid}.h5', 'r')['shot'][()]

    return signal1, signal2, np.asarray(shots[:,0].astype(int))

### MAIN ###

shotnum = 178631

## --- Load ---
file_ids = np.genfromtxt('/projects/EKOLEMEN/agarcia/fid.txt', dtype=int)
i = (file_ids<=shotnum).argmin()
fid = file_ids[i]

signal1, _, _ = load_time_series_tensor(fid, 'v1', 'v3')
_, signal2, signal_shots = load_time_series_tensor(fid, 'v2', 'r0')

## --- Calculate ---
i = abs(signal_shots-shotnum).argmin()
t = h5.File('/projects/EKOLEMEN/agarcia/time_series/tsignal.h5', 'r')['time'][()]
ampsp, freq, time = ae_co2(signal1[i], signal2[i], t)

## --- Plot ---
fig = plt.figure(figsize=(8,4),dpi=100)
gs = gridspec.GridSpec(2, 1)
ax2 = plt.subplot(gs[:])

ax2.imshow(np.log(ampsp).T,
           origin='lower', cmap='hot', aspect='auto',
           extent=[time.min(), time.max(), freq.min(), freq.max()]);
a = plt.ylabel('Frequency [kHz]')
a = plt.xlabel('Time [ms]')
