#!/usr/bin/env python3
import numpy as np
import scipy as sp
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

### CONVENTIONS AND UNITS
### Time: minutes
### Flavin: AU ('mean' subtracted by 'imBackground', before flat-field correction.
###         This will change.)
### Array start:
###     arrays start at 0 (this is Python, not MATLAB)
###     experiment starts at 0 s
###     data point taken the most far back in the past corresponds to timepoint 0
###     cell that is at the top of the CSV is cell no. 0

## Defines general parameters
sampling_pd = 2.5 # data is sampled every 2.5 minutes in this dataset
  # to increase resolution in the frequency domain,
  # 5 is recommended by VanderPlas (2017)
Fs = 1/sampling_pd

## Loads and cleans up data (no changes to the actual numbers)

# reads data: rows are cells, columns are time points
rawdata = pd.read_csv('Flavinexpostest3_ffcorr_small.csv', header = 0)

# slicing: drops row if any element is NaN or zero
# NaN - cells outside aperture
# zero - because invalid for analysis
rawdata = rawdata.dropna()
rawdata = rawdata[(rawdata != 0).all(1)] 

alldata = rawdata.copy() # I'm doing this because I'll use the columns later

# discards first three columns corresponding to 
# cellID, position, and distance from centre
alldata.drop(alldata.columns[[0,1,2]], axis = 1, inplace = True) 
array_timeseries = alldata.to_numpy(copy = True)

# Defines time axis
l = len(alldata.columns) # number of time points
dtime = np.linspace(0, (l-1)*sampling_pd, l)
  # array of times when measurements were taken
  

## Processes data

# defining high-pass filter
def highpass_filter(data, freq, fs):
    b, a = signal.butter(5, freq, btype = 'highpass', fs = Fs)
    y = signal.lfilter(b, a, data)
    return y

# defining notch filter
def notch_filter(data, freq, fs):
    Q = 30
    b, a = signal.iirnotch(freq, Q, fs)
    y = signal.lfilter(b, a, data)
    return y

# filtering
processed_timeseries = np.copy(array_timeseries)
if True:
    for ii in range(len(array_timeseries)):
        # high-pass filter
        processed_timeseries[ii] = highpass_filter(processed_timeseries[ii],
                freq = 1/250, fs = Fs)

        # notch filter: removes what would otherwise be a sharp peak corresponding to 
        # the image acquisition frequency
        processed_timeseries[ii] = notch_filter(processed_timeseries[ii],
                freq = 0.16, fs = Fs)

# Defines dictionary:
# main data structure to store processed data, derivatives, and attributes
test_data = {}
for ii in range(len(processed_timeseries)): ## ii: each cell
    # Loads info into each sub-dictionary
    test_data[ii] = {
            "id"             : ii,
            "position"       : int(rawdata.iloc[ii][1]),
            "time"           : dtime,                    
            "flavin"         : processed_timeseries[ii]
            }

# Normalise by median absolute deviation
if False:
    for ii in range(len(test_data)):
        median = np.median(test_data[ii]['flavin'])
        mad = stats.median_absolute_deviation(test_data[ii]['flavin'])
        test_data[ii]['flavin'] = np.array([(jj - median)/mad for jj in test_data[ii]['flavin']])

# Removes 'long-term trends' based on how population behaves as a whole.
if True:
    flavin_array = [test_data[ii]['flavin'][jj]
            for ii in range(len(test_data))
            for jj in range(l)]
    flavin_array = np.array(flavin_array).reshape(len(test_data), l)
    position_list = list(range(4,13))
    for position in position_list:
        cells_in_position = [ii for ii in range(len(test_data)) 
            if test_data[ii]['position'] == position]
        flavin_mean = np.mean(flavin_array[cells_in_position], axis = 0)
        for jj in cells_in_position:
            test_data[jj]['flavin'] = test_data[jj]['flavin'] - flavin_mean

## Data visualisation
## (ideally these should be part of an object that wraps everything I've done,
## but this should suffice + save time for now)

# Plot time series
def plot_ts(cell_id, raw = False):
    plt.figure()
    if raw:
        plt.plot(test_data[cell_id]['time'], array_timeseries[cell_id])
    else:
        plt.plot(test_data[cell_id]['time'], test_data[cell_id]['flavin'])
    plt.title('Flavin pipelinerescence of cell %d over time' % cell_id)
    plt.xlabel('Time (min)')
    plt.ylabel('Flavin pipelinerescence (AU)')
    plt.show()

## Prompts
evaluations = []
for jj in range(len(test_data)):
    plt.close()
    print("Cell %d" % jj)
    plot_ts(jj)
    s = input("Score: ")
    evaluations.append(s)
    
with open('Flavinexpostest3_ffcorr_small_OscillationEvals.txt', 'w') as fobj:
    for el in evaluations:
        fobj.write(el)
        fobj.write("\n")

