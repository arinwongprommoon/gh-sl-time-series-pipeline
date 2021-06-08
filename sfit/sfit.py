#!/usr/bin/env python3
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit, minimize
import statistics as st
import matplotlib.pyplot as plt

# Class to score oscillations according to Toner & Grima (2013)
# and do other useful analysis and plotting

class sfit:

    def __init__(self,
                 series, 
                 qpercent = 50,
                 sampling_pd = 2.5,
                 delta_range = np.arange(0.05, 1, 0.05),
                 fit_method = 'Powell',
                 omegahat_method = 'solver'):

        # defining inputs/parameters
        self.series = series # time series
        self.qpercent = qpercent # Q50% or Q99% or whatever else
        self.sampling_pd = sampling_pd # sampling period and frequency
        self.Fs = 1/sampling_pd
        self.delta_range = delta_range
        self.fit_method = fit_method # scipy.optimize methods e.g. 'Powell', 'L-BFGS-B'
        self.omegahat_method = omegahat_method # 'analytic' or 'solver'
        
        # initialising blanks for output variables
        self.omegahat = 0
        self.qxx = 0
        self.period = 0
        self.frq = np.zeros(len(series))
        self.power = np.zeros(len(series))
        self.powerfit = np.zeros(len(series))
        self.opt_out = []
        self.popt = []
        self.fitscore = 9999
        self.linescore = 9999
        self.intercept = 0
   
    def score(self):
    # main thing i want it to do
        
        # fft -> (self.frq, self.power)
        n = len(self.series)
        k = np.arange(n)
        T = n/self.Fs
        self.frq = k/T
        self.frq = self.frq[list(range(n//2))]
        Y = np.fft.fft(self.series)/n
        Y = Y[list(range(n//2))]
        self.power = abs(Y)**2 # squared, consistent with Toner & Grima (2013)

        # define function to be fit
        def func(omega, param):
            '''
            Sfit function, as described by Toner & Grima (2013)
            omega: frequency
            param: array-like object that contains the four parameters in the function
            param[0]: alpha
            param[1]: beta
            param[2]: p
            param[3]: q
            '''
            alpha = param[0]
            beta = param[1]
            p = param[2]
            q = param[3]
            return (alpha + beta*(omega**2))/(p + q*(omega**2) + omega**4)
        
        # define euclidean norm, to be minimised in optimisation
        def eunorm(param):
            '''
            Euclidean norm to be minimised, as described by Toner & Grima (2013)
            param: array-like object that contains the four parameters in the function
            '''
            return np.sqrt(np.sum((func(self.frq, param) - self.power)**2))
        
        # define euclidean norm, fit to flat line
        def eunorm_flatline(intercept):
            '''
            Euclidean norm also to be minimised, but the function is literally y = intercept.
            intercept: what it says on the tin
            '''
            return np.sqrt(np.sum(intercept - self.power)**2)

        # fit numerical power spectrum -> popt
        
        # randomise a bunch of parameter vectors over uniform distributions
        # I know that the bounds here and the ones used by BFGS are different but for some reason it works and if they are the same it doesn't work. Still unsure about the maths that goes into these bounds
        nguesses = 10
        guesses = np.stack((np.random.uniform(0,10,nguesses), np.random.uniform(0,10,nguesses), np.random.uniform(0,10,nguesses), np.random.uniform(-10,10,nguesses)), axis = -1)
        self.popt = np.zeros((1,4))
        self.fitscore = 9999
        for ii in range(len(guesses)):
            if self.fit_method == 'L-BFGS-B':
                results = minimize(eunorm, x0 = guesses[ii], method = self.fit_method, bounds = ((0,10),(0,10),(0,10),(-10,10)))
            else:
                results = minimize(eunorm, x0 = guesses[ii], method = self.fit_method)
            # finds best params (this is a shitty implementation)
            if results.fun < self.fitscore:
                self.fitscore = results.fun
                self.popt = results.x
    
        # fits flat line to power spectrum
        lineresults = minimize(eunorm_flatline, x0 = np.mean(self.power), method = 'BFGS')
        self.linescore = lineresults.fun
        self.intercept = lineresults.x

        xx = np.arange(0, np.amax(self.frq), 0.001)
        self.powerfit = func(xx, self.popt) # higher res

        if self.omegahat_method == 'analytic':
            # computes omegahat analytically
            self.omegahat = np.sqrt((-self.popt[0]+np.sqrt(self.popt[0]**2-self.popt[1]*(self.popt[0]*self.popt[3]-self.popt[1]*self.popt[2])))/self.popt[1])
            smax = func(self.omegahat, self.popt)
        elif self.omegahat_method == 'solver':
            # computes omegahat using solver
            self.opt_out = sp.optimize.fmin_powell(lambda x: -func(x, self.popt), 0, full_output = 1, disp = False)
            self.omegahat = abs(float(self.opt_out[0])) # abs because sometimes it gets the peak in the negative region
            smax = -self.opt_out[1]

        # computes QXX%
        # searches around to find the best delta for the initial values to be fed into the solver, so that the solver actually returns four unique solutions as it should. delta is a guess for how far apart the positive roots are. I'm doing this because there doesn't seem to be one good delta that works for all the experimental data tested so far
        for delta in self.delta_range:
            roots = sp.optimize.fsolve(lambda x: func(x, self.popt) - (self.qpercent/100)*smax, [-self.omegahat-delta, -self.omegahat+delta, self.omegahat-delta, self.omegahat+delta]) # build in the fact that it actually has four roots, but we're only interested in the two positive roots 
            positive_roots = [r for r in roots if r > 0]
            positive_roots = np.asarray(positive_roots) # these couple lines remove 'duplicate' values while being mindful of floating point precision
            _, unique = np.unique(positive_roots.round(decimals=8), return_index = True)
            positive_roots = positive_roots[unique]
            if len(positive_roots) < 2:
                d_omega = -1 # to do: make error handling better than this, use try...except for instance
            else:
                d_omega = abs(positive_roots[0] - positive_roots[1])
            
            if d_omega > 1e-8:
                break
                
        self.qxx = self.omegahat/d_omega
        if self.omegahat == 0:
            self.period = -1 # to do: make error handling better than this, use try...except for instance
        else:
            self.period = 100/self.omegahat

    def draw_timeseries(self):
        xdata = np.arange(len(self.series))
        plt.plot(xdata, sampling_pd*self.series)
        plt.xlabel('Time (min)')
        plt.ylabel('Flavin fluorescence (AU)')
        plt.show()
    
    def draw_powerspectrum(self):
        plt.plot(self.frq, self.power)
        plt.xlabel('Frequency ($min^{-1}$)')
        plt.ylabel('Power')
        plt.show()

    def draw_sfit(self):
        plt.plot(self.frq, self.power)
        #plt.plot(self.frq, self.powerfit) # note that the horizontal axis resolution is the same as the power spectrum, which means that the peak may be missed; a higher resolution would fix this (i've used geogebra to confirm this, trust me)
        xx = np.arange(0, np.amax(self.frq), 0.001)
        plt.plot(xx, self.powerfit)
        plt.xlabel('Frequency ($min^{-1}$)')
        plt.ylabel('Power')
        plt.show()

    def draw_all(self):
        fig, axs = plt.subplots(2, 1)
        xdata = np.arange(len(self.series))
        
        axs[0].plot(self.sampling_pd*xdata, self.series)
        axs[0].set_xlabel('Time (min)')
        axs[0].set_ylabel('Flavin fluorescence (AU)')

        axs[1].plot(self.frq, self.power, color='darkblue', marker='x', linestyle='dashed')
        xx = np.arange(0, np.amax(self.frq), 0.001)
        axs[1].plot(xx, self.powerfit, color='orange')
        #axs[1].plot(self.frq, np.repeat(self.intercept, len(self.frq)))
        axs[1].set_xlabel('Frequency ($min^{-1}$)')
        axs[1].set_ylabel('Power')
        
        plt.show()


