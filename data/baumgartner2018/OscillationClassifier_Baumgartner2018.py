#!/usr/bin/env python3
import numpy as np
import scipy as sp
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt

## Loads and stores time series as a dictionary
## (works in a similar way to the struct in OscillationClassifier.mat)

# Loads and stores time series data
raw = pd.read_csv('Baumgartner2018_AllNormData.csv', header = 0)
#raw = pd.read_csv('Baumgartner2018_AllSmoothStored.csv', header = 0)

# Counts number of columns called 'time_*' and 'flavin_*'
time_ncols = len([s for s in raw.columns.values if "time_" in s])
flavin_ncols = len([s for s in raw.columns.values if "flavin_" in s])

# Loads matrix that stores manually-defined oscillation categories
oscillation_evals = pd.read_csv('Baumgartner2018_OscillationEvals.csv', sep = ',', header = None)

# Reshapes it and removes zeros
dcategory = oscillation_evals.values.flatten()
dcategory = dcategory[np.nonzero(dcategory)]

# Defines dictionary
test_data = {}
for ii in range(len(raw)): ## ii: each cell
    # Creates np arrays to store times and flavin readings
    dtime = np.array(raw.iloc[ii][1:(time_ncols+1)])
    dtime = dtime[~np.isnan(dtime)]
    dflavin = np.array(raw.iloc[ii][(time_ncols+1):(time_ncols+flavin_ncols+1)])
    dflavin = dflavin[~np.isnan(dflavin)]
    # Loads info into each sub-dictionary
    test_data[ii] = {
            "id"        : int(raw.iloc[ii][0]), 
                # Note IDs start from 1 to be consistent with my MATLAB code
                # I am likely to change it later when I start cursing, inevitably
            "time"      : dtime,        # time axis
            "flavin"    : dflavin,      # flavin 
            "category"  : dcategory[ii],# category as I manually defined
            "spec_freqs": None,         # frequency (x) axis for periodogram
            "spec_power": None,         # power (y) axis for periodogram
            "pval"      : None,         # p-value for scoring
            "score"     : None          # algorithm-defined score if oscillation
            }

## Scoring
# -- based on Glynn et al. (2006) and VanderPlas (2017). Here I'm using the classical
# periodogram instead of the LSP because my time series data is evenly-spaced

# Define some parameters
sampling_pd = 5 # data is sampled every 5 minutes in this dataset
oversampling_factor = 5 
  # to increase resolution in the frequency domain,
  # 5 is recommended by VandePlas (2017)

for ii in range(len(test_data)):
    # Defining stuff
    l_ts = test_data[ii]['time'][-1] - test_data[ii]['time'][0] # length of ts
    f_lb = 1/l_ts # lower end of frequency
    f_ub = 0.5 * (1/sampling_pd) # upper end of frequency - Nyquist limit
    
    # Computes the periodogram
    test_data[ii]['spec_freqs'], test_data[ii]['spec_power'] = signal.periodogram(
            test_data[ii]['flavin'], 
            fs = 1/(oversampling_factor*sampling_pd), 
            nfft = (len(test_data[ii]['time']) - 1)*oversampling_factor, 
            detrend = 'constant', 
            return_onesided = True, 
            scaling = 'density')
    test_data[ii]['spec_power'] = test_data[ii]['spec_power'] / oversampling_factor
      # Makes the vertical axis consistent with OscillationClassifier.m
      # i.e. the values of the power spectral density of this classical periodogram
      # are consistent with the power spectral density of the LSP as implemented in
      # the current version of OscillationClassifer.m
      # This should make sense because the LSP converges to a classical periodogram
      # with evenly-spaced time series data

    # Calculates p-value
    M = f_ub * l_ts
    x_g = max(test_data[ii]['spec_power'])
    test_data[ii]['pval'] = 1 - (1 - np.exp(-x_g))**M
                                                   
## Computes results, confusion matrix, and related values
## -- in a function because I'm re-using it
def compute_results(q): # I really need a better name
    # Creates data frame to store results
    result_id = [test_data[ii]['id'] for ii in range(len(test_data))]
    result_category = [test_data[ii]['category'] for ii in range(len(test_data))]
    result_pval = [test_data[ii]['pval'] for ii in range(len(test_data))]
    result = pd.DataFrame(
                np.transpose(np.array([result_id, result_category, result_pval])),
                columns = ['id', 'category', 'pval'])
    # Orders p-values
    result = result.sort_values(by = ['pval'])
    result = result.reset_index(drop = True)

    # Finds k^
    classification = []
    for k in range(len(result)):
        classification.append(result.pval[k] <= q*(k/len(test_data))) # classifier
    result['classification'] = classification
    result.columns = result.columns.get_level_values(0)

    # Generates summary
    # rows = categories [1,2,3]
    # columns = is oscillation [False, True]
    resultsummary = pd.crosstab(result['category'], result['classification'])

    # Generates confusion matrix and associated values
    # Based on only the category 1 and 3 elements
    cm = np.array([[resultsummary.iloc[0,1], resultsummary.iloc[2,1]],
                   [resultsummary.iloc[0,0], resultsummary.iloc[2,0]]])

    # Calculates some ML-related things
    fpr = cm[0,1]/(cm[0,1]+cm[1,1])
    tpr = cm[0,0]/(cm[0,0]+cm[1,0])
    fdr = cm[0,1]/(cm[0,1]+cm[0,0])
    accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])

    # Defines all the stuff I could possibly want out of this in a dictionary
    # because I'm just lazy. This is probably not the most proper thing to do.
    # I probably should be using an object.
    result_dict = {
            "result"       : result,
            "resultsummary": resultsummary,
            "cm"           : cm,
            "fpr"          : fpr,           # false positive rate
            "tpr"          : tpr,           # true positive rate
            "fdr"          : fdr,           # false discovery rate
            "accuracy"     : accuracy
            }
    return result_dict

ROC = True

if ROC:
    # Sweep over a range of q values to create ROC curve
    x = []
    y = []
    l = np.linspace(0.001, 0.999, 1000)
    for q in l:
        r = compute_results(q)
          # I am going to fucking weep when this takes forever
        x.append(r['fpr'])
        y.append(r['tpr'])
    x = np.array(x) # yes, I know I should have defined an empty numpy array at first,
                    # but I remember I used to have trouble with this?
    y = np.array(y)
    
    # Plots ROC curve
    plt.figure()
    plt.plot(x, y)
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.xlim(0,1)
    plt.ylabel('True positive rate')
    plt.ylim(0,1)
    plt.show()

    # Finds best q -- should be closest to (0,1) on the ROC curve
    s = np.sqrt(x**2 + (1 - y)**2)
    bestqs = l[s == min(s)]

    # Use this best q to do the usual analysis
    result_dict = compute_results(bestqs[0])
else:
    result_dict = compute_results(0.10) # put your favourite q here

## p-value plot
xdata = np.arange(len(test_data))
plt.figure()
plt.plot(xdata, result_dict['result'].pval, label = 'p-value')
plt.plot(xdata, result_dict['result'].classification, label = 'classification')
plt.title('p-value plot, for q = %f' % bestqs[0])
plt.xlabel('Cell, arranged by p-value')
plt.ylabel('p-value/classification')
plt.legend()
plt.show()
