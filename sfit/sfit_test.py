#!/usr/bin/env python3
import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd
import statistics as st
import csv
from sfit import sfit

## Define some parameters
sampling_pd = 2.5 # data is sampled every 2.5 minutes in this dataset
Fs = 1/sampling_pd

## Loads and cleans up data (no changes to the actual numbers)

# reads data: rows are cells, columns are time points
rawdata = pd.read_csv('../Flavinexpostest3_ffcorr.csv', header = 0)

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
l = len(alldata.columns)
dtime = np.linspace(0, (l-1)*sampling_pd, l)

## Processes data

# defining high-pass filter - can probably collapse into one function tbh
def matlab_highpass_filter(data, freq, fs):
    b, a = signal.butter(5, freq, btype = 'highpass', fs = Fs)
    y = signal.lfilter(b, a, data)
    return y

# defining notch filter - can probably collapse into one function
def matlab_notch_filter(data, freq, fs):
    Q = 30
    b, a = signal.iirnotch(freq, Q, fs)
    y = signal.lfilter(b, a, data)
    return y

# detrending and filtering
processed_timeseries = np.zeros_like(array_timeseries)
for ii in range(len(array_timeseries)):
    # detrend with Nth order polynomial
    N = 1
    model = np.polyfit(dtime, array_timeseries[ii], N)
    predicted = np.polyval(model, dtime)
    processed_timeseries[ii] = array_timeseries[ii] - predicted

    # high-pass filter: lets in frequencies greater than the one corresponding to
    # twice the length of time series (hard-coded for now)
    processed_timeseries[ii] = matlab_highpass_filter(processed_timeseries[ii],
            freq = 0.0005571, fs = Fs)

    # notch filter: removes what would otherwise be a sharp peak corresponding to 
    # the image acquisition frequency
    processed_timeseries[ii] = matlab_notch_filter(processed_timeseries[ii],
            freq = 0.16, fs = Fs)

# scoring and writing optimal parameters to csv
analysis = [] # each element of this is each cell and associated functions

with open('OptimalParameters.csv','w') as fobj:
    writer = csv.writer()
    for ii in range(len(array_timeseries)):
        analysis.append(sfit(series = processed_timeseries[ii], sampling_pd = sampling_pd, qpercent = 50))
        analysis[ii].score()
        print("Time series", ii, "scored.")
        row = [ii, np.ptp(processed_timeseries[ii]), analysis[ii].omegahat, analysis[ii].qxx, *analysis[ii].popt]
        writer.writerow(row)
