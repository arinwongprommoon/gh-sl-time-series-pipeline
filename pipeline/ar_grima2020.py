#!/usr/bin/env python3
import numpy as np
import scipy as sp
import pandas as pd
import itertools
import matplotlib.pyplot as plt

import scipy.linalg as linalg
from statsmodels.tsa.arima_process import ArmaProcess

class AR_Fit:
    """
    Main point of this is to get the AIC as a function of my timeseries and a
    specified AR order.  Is a class so that things are a bit better-structured
    """
    def __init__(self, timeseries, ar_order):
        self.timeseries = timeseries
        self.ar_order = ar_order
        # mean of timeseries, represented by <n> in publication
        self.mean = np.mean(timeseries)
        # length of timeseries, represented by M in publication
        self.length = len(timeseries)
        # sample autocorrelation function, represented by R in publication
        self.sample_acfs = self.get_sample_acf()
        # autoregression coefficients, represented by phi in publication
        # ar_coeffs[0] is always 1, consistent with definition in publication
        self.ar_coeffs = self.get_ar_coeffs()
        # noise parameter, represented by theta-nought-squared in publication
        self.noise_param = self.get_noise_param()
        # Akaike information criterion (AIC)
        self.aic = self.get_aic()

    def get_sample_acf(self):
        """
        Estimates sample autocorrelation function (R).  Returns 1D array of R
        values.
        """
        # initialise
        sample_acfs = np.zeros(self.ar_order+1)
        # summation
        for ii in range(self.ar_order+1):
            sample_acfs[ii] = (1/self.length) * \
                np.sum([(self.timeseries[k] - self.mean) * \
                        (self.timeseries[k+ii] - self.mean)
                        for k in range(self.length - ii)])
        return sample_acfs

    def get_ar_coeffs(self):
        """
        Estimates AR coefficients (phi) by solving Yule-Walker equation.
        Returns 1D array of coefficients (i.e. phi values)
        """
        sample_acfs_toeplitz = linalg.toeplitz(self.sample_acfs[0:self.ar_order])
        # phi vector goes from 1 to P in the publication...
        ar_coeffs = \
                linalg.inv(sample_acfs_toeplitz).dot(self.sample_acfs[1:self.ar_order+1])
        # defines a dummy phi_0 as 1.  This is so that the indices I use in
        # get_noise_param are consistent with the publication.
        ar_coeffs = np.insert(ar_coeffs, 0, 1., axis = 0)
        return ar_coeffs

    def get_noise_param(self):
        """
        Estimates noise parameter
        """
        return self.sample_acfs[0] - \
            np.sum([self.ar_coeffs[k] * self.sample_acfs[k]
                    for k in range(1, self.ar_order+1)])

    def get_aic(self):
        """
        Calculates AIC
        """
        return np.log(self.noise_param) + (2*self.ar_order)/self.length

    def simulate_ar_process(self):
        """
        Outputs simulated AR processed based on the coefficients found
        """
        # This could be its own function, outside the AR_Fit class.
        # Currently it's sort of a bodge so I can get the fitted model and
        # compare it with the time series


def optimise_ar_order(timeseries, ar_order_upper_limit):
    """
    Finds the optimal order P of the AR(P) model by minimising AIC.
    """
    # Bug: artificial dip at order 1 if time series is a smooth sinusoid.
    # Will probably need to fix it so that it checks if the minimum also
    # corresponds to a zero derivative.
    ar_orders = np.arange(1, ar_order_upper_limit)
    aics = np.zeros(len(ar_orders))
    for ii, ar_order in enumerate(ar_orders):
        model = AR_Fit(timeseries, ar_order)
        aics[ii] = model.aic
    return ar_orders[np.argmin(aics)]
    # alternative/another function: pass the AR_Fit object to an AR_Power call?
    # the way this is used in pipeline.periodogram.autoreg() right now is
    # redundant...

class AR_Power:
    """
    Contains AR model parameters, frequency axis, and estimated sample power
    spectrum via a closed-form formula
    """
    def __init__(self, ar_model, freqs, normalise):
        self.ar_model = ar_model
        self.freqs = freqs
        self.power = self.estimate_ps(normalise)

    def estimate_ps(self, normalise = False):
        # logically the condition of whether I should normalise should be in
        # __init__ but I'll fix it later.  Doesn't get in the way of
        # functionality.
        """
        Estimates power spectrum
        """
        power = np.zeros(len(self.freqs))
        for ii, freq in enumerate(self.freqs): # xi
            # multiplied 2pi into the exponential to get the frequency rather
            # than angular frequency
            summation = [-self.ar_model.ar_coeffs[k] * \
                         np.exp(-1j * k * (2*np.pi) * freq)
                       for k in range(self.ar_model.ar_order+1)]
            summation[0] = 1 # minus sign error???
            divisor = np.sum(summation)
            power[ii] = (self.ar_model.noise_param / (2*np.pi)) / \
                    np.power(np.abs(divisor), 2)
        # Normalise by first element of power axis.  This is consistent with
        # the MATLAB code.  Don't know why they do it.  What I'd like to do is
        # make the area under the curve constant, but I'll have to check
        # whether simple normalisation in this way makes the area under the
        # curve constant (checking numbers & arithmetic...)
        if normalise:
            power = power / power[0]
        return power
