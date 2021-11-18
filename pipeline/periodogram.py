#!/usr/bin/env python3
import numpy as np
import scipy.signal as signal

from . import CellAttr

import pipeline.ar_grima2020 as ar_grima2020

def classical(t,
        y,
        oversampling_factor = 1):
    """
    Computes classical periodogram, normalised by variance.

    Parameters:
    -----------
    t = 1D array-like
        time axis
    y = 1D array-like
        measurement
    oversampling_factor = integer
        oversampling factor

    Return:
    -------
    freqs = 1D array-like
        frequency axis
    power = 1D array-like
        power axis, dimensionless, normalised by variance
    """
    l = len(t)
    sampling_pd = t[1] - t[0]
    # Computes periodogram
    # (defines new properties for class)
    freqs, power = signal.periodogram(
            y,
            fs = 1/(oversampling_factor*sampling_pd),
            nfft = (l)*oversampling_factor,
            detrend = 'constant',
            return_onesided = True,
            scaling = 'spectrum')
    freqs = oversampling_factor * freqs
      # multiplies back the oversampling factor so that the units are
      # expressed in min-1
    power = power * (0.5*l)
      # multiply by half the number of time points.  This is consistent with
      # Scargle (1982) and Glynn et al. (2006)
    power = power / np.var(y, ddof = 1)
      # normalise by the variance - this is done in Scargle (1982), and it also
      # allows comparing different time series
    return freqs, power

def bgls(t, y, err, plow= 0.5, phigh= 100, ofac= 10):
    """
    Computes Bayesian General Lomb-Scargle according to Mortier et al. (2015)

    Copyright (c) 2014-2015 Annelies Mortier, João Faria
    with some of my tweaks
    from BGLS: A Bayesian formalism for the generalised Lomb-Scargle periodogram
    by A Mortier, J P Faria, C M Correia, A Santerne, and N C Santos

    Parameters:
    -----------
    t = 1D array-like
        time axis
    y = 1D array-like
        measurements
    plow = float
        lower bound of period
    phigh = float
        upper bound of period
    ofac = integer
        oversampling frequency

    Return:
    -------
    f = 1D numpy array
        frequency axis
    p = 1D numpy array
        probabilities, normalised so that highest is 1
    """
    eps= np.finfo(float).eps
    n_steps= int(ofac*len(t)*(1/plow - 1/phigh))
    f= np.linspace(1/phigh, 1/plow, n_steps)
    omegas= 2*np.pi*f
    err2= err*err
    w= 1/err2
    W= sum(w)
    bigY= sum(w*y)  # Eq. (10)
    p, constants, exponents = [], [], []

    for i, omega in enumerate(omegas):
        theta= 0.5*np.arctan2(sum(w*np.sin(2.*omega*t)), sum(w*np.cos(2.*omega*t)))
        x= omega*t - theta
        cosx= np.cos(x)
        sinx= np.sin(x)
        wcosx= w*cosx
        wsinx= w*sinx
        C= sum(wcosx)
        S= sum(wsinx)
        YCh= sum(y*wcosx)
        YSh= sum(y*wsinx)
        CCh= sum(wcosx*cosx)
        SSh= sum(wsinx*sinx)
        if np.abs(CCh) > eps and np.abs(SSh) > eps:
            K= (C*C*SSh + S*S*CCh - W*CCh*SSh)/(2*CCh*SSh)
            L= (bigY*CCh*SSh - C*YCh*SSh - S*YSh*CCh)/(CCh*SSh)
            M= (YCh*YCh*SSh + YSh*YSh*CCh)/(2*CCh*SSh)
            constants.append(1/np.sqrt(CCh*SSh*abs(K)))
        elif np.abs(CCh) <= eps:
            K= (S*S - W*SSh)/(2*SSh)
            L= (bigY*SSh - S*YSh)/(SSh)
            M= (YSh*YSh)/(2*SSh)
            constants.append(1/np.sqrt(SSh*abs(K)))
        elif np.abs(SSh) <= eps:
            K= (C*C - W*CCh)/(2*CCh)
            L= (bigY*CCh - C*YCh)/(CCh)
            M= (YCh*YCh)/(2*CCh)
            constants.append(1/np.sqrt(CCh*abs(K)))
        if K > 0:
            raise RuntimeError('K is positive. This should not happen.')
        else:
            exponents.append(M - L*L/(4*K))
    constants= np.array(constants)
    exponents= np.array(exponents)
    logp= np.log10(constants) + exponents*np.log10(np.exp(1))
    p= 10**(logp - np.max(logp))
    p= np.array(p)/max(p)  # normalize

    # originally returns 1/f, p but changed so consistent with classical
    return f, p

def autoreg(t,
            y,
            freq_npoints = 100):
    optimal_ar_order = ar_grima2020.optimise_ar_order(y, int(3*np.sqrt(len(y))))
    # out of curiosity
    #print(optimal_ar_order)
    optimal_model = ar_grima2020.AR_Fit(y, optimal_ar_order) # is it doing something redundant?  Should check to see if I can improve performance

    sampling_pd = t[1] - t[0]
    freqs = np.linspace(0, 1/(2*sampling_pd), freq_npoints)

    model_ps = ar_grima2020.AR_Power(optimal_model, freqs, normalise=True)
    # autoreg code doesn't recognise sampling period, added this as corection
    return optimal_ar_order, (1/sampling_pd)*model_ps.freqs, model_ps.power
