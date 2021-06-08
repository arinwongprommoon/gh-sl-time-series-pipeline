#!/usr/bin/env python3
import numpy as np
import scipy as sp
import scipy.signal as signal

from . import rsetattr, rgetattr

# defining high-pass filter
def highpass_filter(data, freq, fs):
    b, a = signal.butter(5, freq, btype = 'highpass', fs = fs)
    y = signal.filtfilt(b, a, data)
      # filter applied twice so that there is no phase shift.
      # important becuase i am overlaying birth times on the flavin time series
    return y

# defining notch filter
def notch_filter(data, freq, fs):
    Q = 30
    b, a = signal.iirnotch(freq, Q, fs)
    y = signal.filtfilt(b, a, data)
    return y

def stdfilter(
        timeseries,
        Fs,
        highpass_cut = 1/360.0,
        notch_freq = 0.16
        ):
    """
    Apply standard filters to cellular time series data

    Parameters:
    -----------
    timeseries: 1D array-like
        time series data, length corresponds to number of time points
    Fs: scalar
        sampling frequency
    highpass_cut: scalar, optional
        frequency for threshold of high pass filter, default 1/360.0 because
        6 hours is a reasonable upper bound for cell cycle lengths
    notch_freq: scalar, optional
        frequency for notch filter, default 0.16 for sampling frequency of
        2.5 min
    """
    # high-pass filter
    timeseries = highpass_filter(timeseries, freq = 1/360.0, fs = Fs)

    # notch filter: removes what would otherwise be a sharp peak corresponding to 
    # the image acquisition frequency
    timeseries = notch_filter(timeseries, freq = 0.16, fs = Fs)

    return timeseries

def population_detrend(
        list_CellAttr,
        cell_attr,
        ):
    """
    Subtracts each time series by population-wise mean in each position

    Parameters:
    -----------
    list_CellAttr: list of CellAttr objects
    cell_attr: string,
        CellAttr attribute to get the data from
    """
    # assumes all ts have equal lengths
    l = len(rgetattr(list_CellAttr[0], cell_attr))
    # creates 2d array: rows are cells, columns are measurements at each
    # time point
    fluo_array = [rgetattr(list_CellAttr[ii], cell_attr)[jj]
            for ii in range(len(list_CellAttr))
            for jj in range(l)]
    fluo_array = np.array(fluo_array).reshape(len(list_CellAttr), l)
    # for each position present in this experiment...
    for position in np.unique([cell.position for cell in list_CellAttr]):
        # cellids of cells in position of interest
        cells_in_position = [ii for ii in range(len(list_CellAttr))
                if list_CellAttr[ii].position == position]
        # calculate mean, population-wise
        fluo_mean = np.mean(fluo_array[cells_in_position], axis = 0)
        # subtracts measure by this mean -- so that I get rid of any position-
        # specific, population-wide long-term trend in the data that doesn't
        # provide much information
        for jj in cells_in_position:
            temp = rgetattr(list_CellAttr[jj], cell_attr)
            rsetattr(list_CellAttr[jj], cell_attr, temp - fluo_mean)
