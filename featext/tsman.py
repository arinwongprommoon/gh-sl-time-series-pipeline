#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.stats as st

def UnityRescale(x):
    """
    Linearly rescale a data vector to unit interval
    (adapted from hctsa)
    """
    goodVals = ~np.isnan(x)
    if ~np.any(goodVals): # all NaNs
        xhat = x
    minX = min(x[goodVals])
    maxX = max(x[goodVals])
    if minX == maxX:
        # There is no variabtion -- set all to zero -- attempts to rescale will
        # blow up
        xhat = x
        xhat[goodVals] = 0
    else:
        # There is some variation -- rescale to unit interval
        xhat = (x-minX)/(maxX-minX)
    return xhat

def Sigmoid(x):
    """
    Classic sigmoidal transformation, scaled to the unit interval
    (adapted from hctsa)
    """
    goodVals = ~np.isnan(x)
    meanX = np.mean(x[goodVals])
    stdX = np.std(x[goodVals])

    # Sigmoidal transformation
    xhat = 1/(1 + np.exp(-(x-meanX)/stdX))
    # Rescale to unit interval
    xhat = UnityRescale(xhat)
    return xhat

def RobustSigmoid(x):
    """
    Outlier-adjusted sigmoid, scaled to the unit interval
    (adapted from hctsa)
    """
    goodVals = ~np.isnan(x)
    medianX = np.median(x[goodVals])
    iqrX = st.iqr(x[goodVals])

    if iqrX == 0: # Can't apply an outlier-robust sigmoid meaningfully
        xhat = np.empty(x.shape)
        xhat[:] = np.nan
    else:
        # Outlier-robust sigmoid
        xhat = 1/(1 + np.exp(-(x-medianX)/(iqrX/1.35)))
        xhat = UnityRescale(xhat)
    return xhat

def TS_Normalize(dataMatrix):
    """
    Normalises an hctsa data matrix (a feature matrix) to values between 0 and 1
    using the mixedSigmoid method.
    (adapted from hctsa)

    Parameters:
    -----------
    dataMatrix = 2D numpy array

    Return: dataMatrixNorm = 2D numpy array, normalised data matrix
    """
    dataMatrixNorm = np.zeros(dataMatrix.shape)
    for ii in range(dataMatrix.shape[1]):
        if max(dataMatrix[:,ii]) == min(dataMatrix[:,ii]):
            # A constant column is set to 0
            dataMatrixNorm[:,ii] = 0
        elif sum(np.isnan(dataMatrix[:,ii])):
            # Everything a NA, kept at NA
            dataMatrixNorm[:,ii] = np.nan
        elif st.iqr([dataMatrix[jj,ii] for jj in range(len(dataMatrix))]) == 0:
            # iqr of data is zero: perform a normal sigmoidal transformation
            dataMatrixNorm[:,ii] = Sigmoid(dataMatrix[:,ii])
        else:
            # Perform an outlier-robust version of the sigmoid
            dataMatrixNorm[:,ii] = RobustSigmoid(dataMatrix[:,ii])
    return dataMatrixNorm
