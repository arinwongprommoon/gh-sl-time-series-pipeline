#!/usr/bin/env python3
import numpy as np
import scipy.optimize as op
from scipy.special import comb
from scipy.optimize import curve_fit, minimize, fmin_powell

from . import PdgramAttr, ScoreAttr, CellAttr

def glynn06(cell, pdgram = None):
    """
    Score according to Glynn et al. (2006) (prototype)

    Parameters:
    -----------
    cell = pipeline.CellAttr object

    Return:
    -------
    out = pipeline.ScoreAttr object
    """
    if pdgram is None:
        pdgram = cell.flavin.classical

    out = ScoreAttr()
    out.label = 'Xg'

    out.score = max(pdgram.power)

    return out

def wichert03(cell, pdgram = None):
    """
    Score according to Wichert et al. (2003)

    Parameters:
    -----------
    cell = pipeline.CellAttr object

    Return:
    -------
    out = pipeline.ScoreAttr object
    """
    # default
    if pdgram is None:
        pdgram = cell.classical

    out = ScoreAttr()
    out.label = 'p-value'

    g = max(pdgram.power) / np.sum(pdgram.power)
    l = len(cell.time)
    n = l//2
    p = int(np.floor(1/g))
    pval = 0
    for jj in range(1, p+1):
        pval += (-(-1)**jj) * comb(n, jj) * ((1 - jj*g)**(n - 1))

    out.score = pval

    return out

def sfit(cell, pdgram = None):

    if pdgram is None:
        pdgram = cell.classical

    out = ScoreAttr()
    out.label = 'Q50%'

    def model(omega, param):
        '''mathematical description of function to fit periodogram to'''
        # these 4 lines here purely to make code readable
        alpha = param[0]
        beta = param[1]
        p = param[2]
        q = param[3]
        return (alpha + beta*(omega**2))/(p + q*(omega**2) + omega**4)

    # hard-coded so that i don't have to faff around with lambdas when i use
    # op.minimize; intent is to vary param until Euclidean norm is at minimum
    def eunorm(param):
        '''Euclidean norm'''
        return np.sqrt(np.sum(model(pdgram.freqs, param) - pdgram.power)**2)

    # generates initial guesses of sfit model parameter values (alpha, beta,
    # p, q).  number of guesses specified by nguesses.
    nguesses = 100

    class InitialGuess:
        '''single-use class to contain stuff about initial guesses'''
        def __init__(self):
            # low/high arguments are constraints to sfit model parameter values
            self.param_initial = [np.random.uniform(0,1),
                                  np.random.uniform(0,1),
                                  np.random.uniform(0,1),
                                  np.random.uniform(-1,1)]
            self.fitresults = op.minimize(eunorm,
                                          x0 = self.param_initial,
                                          method = 'Powell')

    guesses = [InitialGuess() for ii in range(nguesses)]

    # identifies best guess and gets optimal parameters
    fitscores = np.array([guess.fitresults.fun for guess in guesses])
    best_guess_idx = np.argmin(fitscores)
    param_optimal = guesses[best_guess_idx].fitresults.x

    # finds omegahat, smax
    solver_results = op.fmin_powell(lambda x: -model(x, param_optimal),
                                    0, full_output = 1, disp = False)
    omegahat = abs(float(solver_results[0]))
    smax = -solver_results[1]

    # computes QXX%
    qpercent = 50
    # delta is a guess for how far apart the positive roots are, somehow using
    # this makes the solver better.  here it searches around to find the best
    # delta for initial values.  i do this because there doesn't seem to be one
    # good delta that works for all
    delta_range = np.arange(0.0001, 0.2, 0.0001)
    for delta in delta_range:
        # maths: there are four roots, but we are only interested in the
        # two positive ones
        roots = op.fsolve(lambda x: model(x, param_optimal) - (qpercent/100)*smax,
                          [-omegahat - delta,
                           -omegahat + delta,
                            omegahat - delta,
                            omegahat + delta])
        positive_roots = np.array([r for r in roots if r > 0])
        # removes 'duplicates' to eight-decimal precision
        _, unique = np.unique(positive_roots.round(decimals = 8),
                              return_index = True)
        positive_roots = positive_roots[unique]
        # if there are < 2 positive roots, move on to next delta in delta_range
        if len(positive_roots) < 2:
            d_omega = 0 # very rudimentary error handling
        else:
            d_omega = abs(positive_roots[0] - positive_roots[1])

        # when d_omega is non-zero (to eight-decimal precision),
        # it's good enough
        if ~np.isclose(d_omega, 0):
            break

    qxx = omegahat / d_omega

    out.score = qxx

    out.misc.model = model
    out.misc.param_optimal = param_optimal
    out.misc.omegahat = omegahat

    return out

def entropy(cell, pdgram = None):
    """
    Compute spectral entropy, i.e. assumes periodogram is probability
    distribution and calculates Shannon entropy from it

    Parameters:
    -----------
    cell = pipeline.CellAttr object

    Return:
    -------
    out = pipeline.ScoreAttr object
    """
    if pdgram is None:
        pdgram = cell.classical

    out = ScoreAttr()
    out.label = 'spectral entropy'

    # gets sampling period assuming all time points spaced equally
    sampling_pd = cell.time[1] - cell.time[0]

    # entropy of ensemble.  I don't know why, but the maths works this way
    p = 2 * sampling_pd * pdgram.power

    entropy = 0
    for jj in range(len(p)):
        entropy += -p[jj] * np.log2(p[jj])
    entropy = entropy/np.log2(len(pdgram.freqs))

    out.score = entropy

    return out
