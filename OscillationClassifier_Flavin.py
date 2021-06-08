#!/usr/bin/env python3
import numpy as np
import scipy as sp
import scipy.signal as signal
import scipy.stats as stats
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
oversampling_factor = 1 
  # to increase resolution in the frequency domain,
  # 5 is recommended by VanderPlas (2017)
Fs = 1/sampling_pd

## Defines some options -- here so that it's easy to find
filtering = True
population_detrend = True
diagnostic_series = False
ROC = True
my_q = 2.0e-14

## Loads and cleans up data (no changes to the actual numbers)

# reads data: rows are cells, columns are time points
rawdata = pd.read_csv('Flavinexpostest1_subtracted.csv', header = 0)

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


## Loads manually-defined oscillation categories and stores in dcategory
dcategory = []
with open('Flavinexpostest3_ffcorr_OscillationEvals.txt', 'r') as fobj:
    c = fobj.readlines()
    dcategory = [int(x.strip()) for x in c]

## Loads birth times
#birthsdata = pd.read_csv('Flavinexpostest3_ffcorr_small_births.csv', header = 0)

## Processes data

# defining high-pass filter
def highpass_filter(data, freq, fs):
    b, a = signal.butter(5, freq, btype = 'highpass', fs = Fs)
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

# filtering
processed_timeseries = np.copy(array_timeseries)
if filtering:
    for ii in range(len(array_timeseries)):
        # high-pass filter
        processed_timeseries[ii] = highpass_filter(processed_timeseries[ii],
                freq = 1/360.0, fs = Fs)

        # notch filter: removes what would otherwise be a sharp peak corresponding to 
        # the image acquisition frequency
        processed_timeseries[ii] = notch_filter(processed_timeseries[ii],
                freq = 0.16, fs = Fs)

# Defines dictionary:
# main data structure to store processed data, derivatives, and attributes
test_data = {}
for ii in range(len(processed_timeseries)): ## ii: each cell
    position = int(rawdata.iloc[ii][1])
    exposure = 60 * np.floor((position - 1) / 3)
    # Loads info into each sub-dictionary
    test_data[ii] = {
            "id"             : ii,
            "MATLABid"       : int(rawdata.iloc[ii][0]), # cellID in case I need to cross-check with MATLAB
            "position"       : int(rawdata.iloc[ii][1]),
            "distfromcentre" : rawdata.iloc[ii][2],
            "time"           : dtime,                    # time axis
            "flavin"         : processed_timeseries[ii],
            "category"       : dcategory[ii],            # category as I manually defined
            #"category"       : np.random.randint(1, 4),  # until I actually classify my bliddy data
            "births"         : None,
            "spec_freqs"     : None,                     # frequency (x) axis for periodogram
            "spec_power"     : None,                     # power (y) axis for periodogram
            "pval"           : None,                     # p-value for scoring
            "score"          : None                      # algorithm-defined score if oscillation
            }

# Loads birth times into sub-dictionaries, cross-referencing MATLABid
#for ii in range(len(test_data)): # For each cell in test data...
#    # ...gets MATLABid of the cell, and looks for the entry with this id in
#    # birthsdata.  Works because birthdata's 'id' column runs from 1 to 1129
#    # without gaps; the -1 is to correct for OBOE
#    dbirths_temp = birthsdata.iloc[int(test_data[ii]['MATLABid'] - 1)]
#    dbirths = np.array(dbirths_temp.dropna()) # drops NaNs
#    test_data[ii]['births'] = dbirths[1:] # drops first element which corresponds to id

# Removes 'long-term trends' based on how population behaves as a whole.
if population_detrend:
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

# Plot mean +/- SD
if True:
    flavin_array = [test_data[ii]['flavin'][jj]
            for ii in range(len(test_data))
            for jj in range(l)]
    flavin_array = np.array(flavin_array).reshape(len(test_data), l)
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    colour = 'cornflowerblue'
    position_list = list(range(4,13))
    for position in position_list:
        cells_in_position = [ii for ii in range(len(test_data)) 
            if test_data[ii]['position'] == position]
        flavin_mean = np.mean(flavin_array[cells_in_position], axis = 0)
        flavin_std = np.std(flavin_array[cells_in_position], axis = 0)
        axs[(position-4)//3, (position-4)%3].plot(dtime, flavin_mean, color = colour)
        axs[(position-4)//3, (position-4)%3].plot(dtime, np.subtract(flavin_mean, flavin_std), color = colour, linestyle = ':')
        axs[(position-4)//3, (position-4)%3].plot(dtime, np.add(flavin_mean, flavin_std), color = colour, linestyle = ':')
        axs[(position-4)//3, (position-4)%3].set_ylim(0, 200)
        axs[(position-4)//3, (position-4)%3].title.set_text('Position %d' % position)
    fig.suptitle('Mean flavin pipelinerescence within each position (+/- 1 SD)')
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False,
            right = False, grid_alpha = 0)
    plt.xlabel('Time (min)')
    plt.ylabel('Flavin pipelinerescence (AU)')
    fig.show()

# Plot all ts
if True:
    flavin_array = [test_data[ii]['flavin'][jj]
            for ii in range(len(test_data))
            for jj in range(l)]
    flavin_array = np.array(flavin_array).reshape(len(test_data), l)
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    colour = 'cornflowerblue'
    position_list = list(range(4,13))
    for position in position_list:
        cells_in_position = [ii for ii in range(len(test_data)) 
            if test_data[ii]['position'] == position]
        flavin_mean = np.mean(flavin_array[cells_in_position], axis = 0)
        flavin_std = np.std(flavin_array[cells_in_position], axis = 0)
        for el in cells_in_position:
            axs[(position-4)//3, (position-4)%3].plot(dtime, test_data[el]['flavin'])
        axs[(position-4)//3, (position-4)%3].set_ylim(0, 200)
        axs[(position-4)//3, (position-4)%3].title.set_text('Position %d' % position)
    fig.suptitle('Flavin pipelinerescence within each position')
    fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False,
            right = False, grid_alpha = 0)
    plt.xlabel('Time (min)')
    plt.ylabel('Flavin pipelinerescence (AU)')
    fig.show()

## Add diagnostic plots
if diagnostic_series:
    # pure sine
    test_data[243] = test_data[0].copy()
    test_data[243]['id'] = 243
    test_data[243]['MATLABid'] = 9990
    test_data[243]['position'] = 99
    test_data[243]['distfromcentre'] = 999
    test_data[243]['time'] = dtime
    test_data[243]['flavin'] = np.sin(2 * np.pi * 0.01667 * dtime)
    test_data[243]['category'] = 9
    # white noise
    test_data[244] = test_data[0].copy()
    test_data[244]['id'] = 244
    test_data[244]['MATLABid'] = 9991
    test_data[244]['position'] = 99
    test_data[244]['distfromcentre'] = 999
    test_data[244]['time'] = dtime
    test_data[244]['flavin'] = np.random.normal(0, 100, size=l)
    test_data[244]['category'] = 9
    # sine + noise
    test_data[245] = test_data[0].copy()
    test_data[245]['id'] = 245
    test_data[245]['MATLABid'] = 9992
    test_data[245]['position'] = 99
    test_data[245]['distfromcentre'] = 999
    test_data[245]['time'] = dtime
    test_data[245]['flavin'] = 30 + \
        -np.cos((2 * np.pi * dtime)/1000) + \
        np.cos((2 * np.pi * dtime)/120) + \
        0.5*np.random.normal(0, 1, size=l)
    test_data[245]['category'] = 9

## Scoring
# -- based on Glynn et al. (2006) and VanderPlas (2017). Here I'm using the classical
# periodogram instead of the LSP because my time series data is evenly-spaced

# Defining key time/frequency values (outside loop because same for all ts)
l_ts = test_data[ii]['time'][-1] - test_data[ii]['time'][0] # duration of ts
f_lb = 1/l_ts # lower end of frequency
f_ub = 0.5 * (1/sampling_pd) # upper end of frequency - Nyquist limit
M = f_ub * l_ts # VanderPlas (2018)

for ii in range(len(test_data)):
    # Computes periodogram
    test_data[ii]['spec_freqs'], test_data[ii]['spec_power'] = signal.periodogram(
            test_data[ii]['flavin'], 
            fs = 1/(oversampling_factor*sampling_pd), 
            nfft = (len(test_data[ii]['time']))*oversampling_factor, 
            detrend = 'constant', 
            return_onesided = True, 
            scaling = 'spectrum')
    test_data[ii]['spec_freqs'] = oversampling_factor * test_data[ii]['spec_freqs']
      # multiplies back the oversampling factor so that the units are
      # expressed in min-1
    test_data[ii]['spec_power'] = test_data[ii]['spec_power'] * (0.5*l)
      # multiply by half the number of time points.  This is consistent with
      # Scargle (1982) and Glynn et al. (2006)
    test_data[ii]['spec_power'] = test_data[ii]['spec_power'] / np.var(test_data[ii]['flavin'], ddof = 1)
      # normalise by the variance - this is done in Scargle (1982), and it also
      # allows comparing different time series
    
    # Gets height of highest 'peak' in periodogram.
    # Glynn et al. (2006) describe calculating a p-value, but I'm doing this instead
    # and modifying the classifier accordingly because the p-values would otherwise
    # be too small.
    x_g = max(test_data[ii]['spec_power'])
    test_data[ii]['pval'] = x_g


## Computes results, confusion matrix, and related values
## -- in a function because I'm re-using it
def compute_results(q): # I really need a better name
    # Creates data frame to store results.
    # This magnificient block of code is why I love Python
    
    # Set which fields of test_data to be used as columns
    result_fields = ['id', 'MATLABid', 'distfromcentre', 'category', 'pval']
    # Creates long-ass list, in which...
    # I go through all test_data elements to get values for each field,
    # make lists from these values, and APPEND these lists to each other.
    # result_templist will be test_data[0]['id'], test_data[1]['id'], ...,
    # test_data[whatever]['id'], ..., test_data[0]['MATLABid'], ... and so on
    result_templist = [test_data[ii][field]
            for field in result_fields
            for ii in range(len(test_data))]
    # And then I reshape the bliddy list into an array
    # (to be transposed in the next line)
    result_temparray = np.array(result_templist).reshape(
            len(result_fields), len(test_data))
    # Puts it into a data frame
    result = pd.DataFrame(np.transpose(result_temparray), columns = result_fields)

    # Orders p-values
    result = result.sort_values(by = ['pval'])
    result = result.reset_index(drop = True)

    # Finds k^
    classification = []
    for k in range(len(result)):
        classification.append(result.pval[k] >= 
                -np.log(1 - (1 - ((q*(k+1))/len(test_data)))**(1/M)))
          # classifier; k+1 to correct OBOE (k starts from 1 in the equations,
          # but starts from 0 because of how Python indexes arrays
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

if ROC:
    # Sweep over a range of q values to create ROC curve
    x = []
    y = []
    lll = np.linspace(-14, -8, 1000)
    ll = np.power(10, lll)
    #ll = np.linspace(10e-15, 1000000e-15, 1000)
    for q in ll:
        r = compute_results(q)
          # I am going to fucking weep when this takes forever
        x.append(r['fpr'])
        y.append(r['tpr'])
    x = np.array(x) # yes, I know I should have defined an empty numpy array at first,
                    # but I remember I used to have trouble with this?
                    # also the reason i'm not using list comprehensions here is
                    # because i would be running compute_results twice; idk i actually
                    # be using a generator or something
    y = np.array(y)

    # Finds best q -- should be closest to (0,1) on the ROC curve
    s = np.sqrt(x**2 + (1 - y)**2)
    bestqs = ll[s == min(s)]

    # Use this best q to do the usual analysis
    result_dict = compute_results(bestqs[0])
else:
    result_dict = compute_results(my_q) # put your favourite q here

## Stores classifications back into test_data
for jj in range(len(result_dict['result'])):
    # I can do all this in one line, but I'm splitting to make my code readable
    cell_id = result_dict['result'].id[jj]
    classification = result_dict['result'].classification[jj]
    test_data[cell_id]['score'] = classification

## Data visualisation
## (ideally these should be part of an object that wraps everything I've done,
## but this should suffice + save time for now)

# Plot time series
def plot_ts(cell_id, births = True, raw = False):
    plt.figure()
    if raw:
        plt.plot(test_data[cell_id]['time'], array_timeseries[cell_id])
    else:
        plt.plot(test_data[cell_id]['time'], test_data[cell_id]['flavin'])
    if births:
        for birth in test_data[cell_id]['births']:
            plt.axvline(birth, ymin = 0, ymax = 1, color = 'r', linestyle = '--')
    else:
        pass
    plt.title('Flavin pipelinerescence of cell %d over time' % cell_id)
    plt.xlabel('Time (min)')
    plt.ylabel('Flavin pipelinerescence (AU)')
    plt.show()

# Save time series
def save_ts(cell_id, name, raw = False):
    plt.figure()
    if raw:
        plt.plot(test_data[cell_id]['time'], array_timeseries[cell_id])
    else:
        plt.plot(test_data[cell_id]['time'], test_data[cell_id]['flavin'])
    plt.title('Flavin pipelinerescence of cell %d over time' % cell_id)
    plt.xlabel('Time (min)')
    plt.ylabel('Flavin pipelinerescence (AU)')
    plt.savefig(str(name) + '.png')

# Plot periodogram
def plot_ps(cell_id, show_period = False):
    plt.figure()
    if show_period:
        plt.plot(1/(test_data[cell_id]['spec_freqs']), 
                test_data[cell_id]['spec_power'])
        plt.xlabel('Period (min)')
        plt.xlim(0, max(test_data[cell_id]['time']))
    else:
        plt.plot(test_data[cell_id]['spec_freqs'], 
                test_data[cell_id]['spec_power'])
        plt.xlabel('Frequency ($min^{-1}$)')
    plt.title('Periodogram (spectrum) of cell %d, with oversampling factor %d' % (cell_id, oversampling_factor))
    plt.ylabel('Power (dimensionless)')
    plt.show()

# Plots ROC curve
def plot_roc():
    plt.figure(figsize=(3,3))
    plt.plot(x, y)
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle = ':')
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.xlim(0,1)
    plt.ylabel('True positive rate')
    plt.ylim(0,1)
    plt.show()

# Kymograph/Heatmap
def kymograph(order_by_score):
    if order_by_score:
        kg_list = [test_data[ii]['flavin'][jj]#/np.std(test_data[ii]['flavin'])
                for ii in result_dict['result'].id # arranged by score
                for jj in range(l)]
    else:
        kg_list = [test_data[ii]['flavin'][jj]
                for ii in range(len(test_data))
                for jj in range(l)]
    kg_array = np.array(kg_list).reshape(len(test_data), l)
    
    seaborn.set()
    plt.figure()
    seaborn.heatmap(kg_array, center = 0, cmap = "RdBu_r")

    # Horizontal axis: time point
    # Vertical axis: cell, 'worst' oscillations on the top, 'best' oscillations bottom
    # Colour: 'flavin', but normalised by standard deviation so I can compare

# Classification plot
def plot_classification():
    xdata = np.arange(len(test_data))
    plt.figure()
    plt.plot(xdata,
            result_dict['result'].pval,
            label = 'Power of highest peak')
    plt.plot(xdata,
            np.max(result_dict['result'].pval) * result_dict['result'].classification,
                # just to scale the classification line for pretty display
            label = 'Classification')
    if ROC:
        plt.title('Classification plot, for q = %e' % bestqs[0])
    else:
        plt.title('Classification plot')
    plt.xlabel('Cell, arranged by power of highest peak in periodogram')
    plt.ylabel('power/classification')
    plt.legend()
    plt.show()
