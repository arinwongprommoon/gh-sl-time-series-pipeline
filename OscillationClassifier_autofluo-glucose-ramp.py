#!/usr/bin/env python3
import copy
import numpy as np
import scipy as sp
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import sklearn.metrics
import igraph as ig

from pipeline import rsetattr, rgetattr
import pipeline.dataexport
import pipeline.dataimport
import pipeline.periodogram
import pipeline.score
import pipeline.tsman
import pipeline.vis

import featext.tsman
import featext.graph
import featext.vis

import catch22
import leidenalg

def add_classicalAttr(cell, oversampling_factor = 1):
    """Computes classical periodogram and adds PdgramAttr attributes"""
    cell.flavin.classical.freqs, cell.flavin.classical.power = \
            pipeline.periodogram.classical(cell.time, cell.flavin.reading_processed,
                                oversampling_factor = oversampling_factor)

def add_bglsAttr(cell):
    """Computes BGLS and adds PdgramAttr attributes"""
    cell.flavin.bgls = pipeline.PdgramAttr()
    cell.flavin.bgls.label = 'Bayesian General Lomb-Scargle Periodogram'
    cell.flavin.bgls.power_label = 'Probability'
    err = np.ones(len(cell.flavin.reading_processed))*\
            np.sqrt(np.max(cell.flavin.reading_processed))
    cell.flavin.bgls.freqs, cell.flavin.bgls.power = \
            pipeline.periodogram.bgls(cell.time, cell.flavin.reading_processed, err,
                    plow = 30.0, phigh = 360.0, ofac = 5)

# specifies the number of time points to drop off the beginning
dropTPs_num = 5
# sampling interval (min)
sampling_interval = 2.5

def dropTPs(df, dropTPs_num):
    df.drop(df.columns[list(range(3, 3+dropTPs_num))], axis = 1, inplace = True)

# Import information from glucose ramp experiment into dataframes
# (flavin_whi5_glucose_limitation_01, 2020-10-28)

ChannelFlavin_rawdata = pipeline.dataimport.import_timeseries( \
        'data/arin/Flavin-whi5-glucose-limitation_Flavin.csv')
# drops first couple time points because of imaging errors
dropTPs(ChannelFlavin_rawdata, dropTPs_num)
# dummy data for oscillation category -- not used for now
ChannelFlavin_dcategory = [3] * len(ChannelFlavin_rawdata)

ChannelmCherry_rawdata = pipeline.dataimport.import_timeseries( \
        'data/arin/Flavin-whi5-glucose-limitation_mCherry.csv')
# drops first couple time points because of imaging errors
dropTPs(ChannelmCherry_rawdata, dropTPs_num)
# dummy data for oscillation category -- not used for now
ChannelmCherry_dcategory = [3] * len(ChannelmCherry_rawdata)
Births = pipeline.dataimport.import_births( \
        'data/arin/Flavin-whi5-glucose-limitation_births.csv')


# Arranges information into DatasetAttr objects
# Reverted to this method rather than using CellAttr_from_datasets because now
# I'm working with different datasets that are formatted differently
# It's fucking awful and I'm not sure how redundant these lines are -- I'm only
# copying old code to do a quick analysis before I go about leveraging
# Alan's exportJSON

GlucoseLimitation_ChannelFlavin_data = pipeline.dataimport.CellAttr_from_datasets( \
        timeseries_df = ChannelFlavin_rawdata,
        categories_array = ChannelFlavin_dcategory,
        births_df = Births)
GlucoseLimitation_ChannelmCherry_data = pipeline.dataimport.CellAttr_from_datasets( \
        timeseries_df = ChannelmCherry_rawdata,
        categories_array = ChannelmCherry_dcategory,
        births_df = Births)
GlucoseLimitation = \
        pipeline.DatasetAttr(GlucoseLimitation_ChannelFlavin_data)
GlucoseLimitation_ChannelmCherry = \
        pipeline.DatasetAttr(GlucoseLimitation_ChannelmCherry_data)
# Add some labels
GlucoseLimitation_ChannelmCherry_MATLABids = \
        [cell.MATLABid for cell in GlucoseLimitation_ChannelmCherry.cells]
for ii, cell in enumerate(GlucoseLimitation.cells):
    cell.source = 'S288c'
    cell.medium.base = 'SC'
    # NOTE TO SELF: future version of code should be able to chop up timeseries
    # according to different nutrient regimes
    cell.medium.nutrients = {'glucose': 0}

    # if I had formatted my CSV to include these values, I wouldn't have to do
    # this FUCKING AWFUL of assigning attribute values

    # Strains
    if cell.position in [1,2,3,4,5,6,7]:
        cell.strain = 'FY4'
    elif cell.position in [9,10,11,12,13,14,15]:
        cell.strain = 'whi5-mCherry'
    # Flavin channel
    cell.flavin = pipeline.Fluo('flavin')
    if cell.position in [1,2,3,4,5,7,9,10,11,12,13,14]:
        cell.flavin.exposure = 120
    elif cell.position in [6,15]:
        cell.flavin.exposure = 0
    cell.flavin.reading = cell.y
    cell.flavin.category = ChannelFlavin_dcategory[ii]
    # mCherry channel
    cell.mCherry = pipeline.Fluo('mCherry')
    if cell.position in [9,10,11,12,13,14]:
        cell.mCherry.exposure = 100
    elif cell.position in [1,2,3,4,5,6,7,15]:
        cell.mCherry.exposure = 0
    # loads in reading, cross-referencing by MATLABid.  This is awful, I know.
    if cell.MATLABid in GlucoseLimitation_ChannelmCherry_MATLABids:
        cell.mCherry.reading = \
                GlucoseLimitation_ChannelmCherry.cells[GlucoseLimitation_ChannelmCherry_MATLABids.index(cell.MATLABid)].y

# Process data
for cell in GlucoseLimitation.cells:
    # apply notch filter
    cell.flavin.reading_processed = pipeline.tsman.notch_filter( \
            cell.flavin.reading, freq = 0.16, fs = 1/sampling_interval)


# Plotting

# Grouping the cells to make life easier
FY4Cells = \
        [cell for cell in GlucoseLimitation.cells if cell.strain == 'FY4']
whi5mCherryCells = \
        [cell for cell in GlucoseLimitation.cells if cell.strain == 'whi5-mCherry']

# Kymographs
if False:
    pipeline.vis.kymograph(FY4Cells, cell_attr = 'flavin.reading_processed')
    pipeline.vis.kymograph(whi5mCherryCells, cell_attr = 'flavin.reading_processed')
    pipeline.vis.kymograph(whi5mCherryCells, cell_attr = 'mCherry.reading')

# Mean cell
# (TODO: add STDEV ranges to plots)
def mean_cell(list_CellAttr, list_attr):
    '''
    Creates an artificial cell that stores the mean time traces of a list of
    specificed attributes (list_attr) across a list of cells (list_CellAttr)
    in its respective attribute.

    And piggybacking on births to store key glucose regime time points, just
    for illustration.
    '''
    MeanCell = copy.deepcopy(list_CellAttr[0])
    for attr in list_attr:
        meants = np.mean([rgetattr(cell, attr) for cell in list_CellAttr],
                        axis = 0)
        rsetattr(MeanCell, attr, meants)
    MeanCell.births = np.array([120, 300, 420, 540]) - \
            dropTPs_num * sampling_interval
    return MeanCell

FY4MeanCell = \
        mean_cell(FY4Cells, \
                  list_attr = ['flavin.reading_processed'])
FY4MeanCell.plot_ts(y_attr = 'flavin.reading_processed', births = True)
whi5mCherryMeanCell = \
        mean_cell(whi5mCherryCells, \
                  list_attr = ['flavin.reading_processed', 'mCherry.reading'])
whi5mCherryMeanCell.plot_ts(y_attr = 'flavin.reading_processed', births = True)
whi5mCherryMeanCell.plot_ts(y_attr = 'mCherry.reading', births = True)

# Example cells
if False:
    FY4Cells[0].plot_ts(y_attr = 'flavin.reading_processed', births = False)
    whi5mCherryCells[0].plot_ts(y_attr = 'flavin.reading_processed', births = False)
    whi5mCherryCells[0].plot_ts(y_attr = 'mCherry.reading', births = False)

# For CSHL 2021 figures
if True:
    class PumpRampRegime:
        def __init__(self, time, glucose_conc):
            self.time = time
            self.glucose_conc = glucose_conc
    autofluo_glucose_ramp_regime = PumpRampRegime([0, 120, 300, 420, 540, 1080],
                                                  [20, 20, 0, 0, 0.5, 0.5])
    fig01_FY4CellAttr = 15
    fig01, axs01 = plt.subplots(nrows = 2, ncols = 1,
                                sharex = True, sharey = False,
                                gridspec_kw = {'hspace': 0.5})
    cell = FY4Cells[fig01_FY4CellAttr]

    axs01[0].plot(cell.time, cell.flavin.reading_processed)
    for event_time in autofluo_glucose_ramp_regime.time:
        axs01[0].axvline(event_time, ymin = 0, ymax = 1,
                         color = 'lightgrey', linestyle = '--')
    axs01[0].set_title(cell.strain + ' strain in ' + cell.medium.base + \
                       ', flavin LED ' + str(cell.flavin.exposure) + ' ms',
                       size = 10)
    axs01[0].set_ylabel('Flavin fluorescence (AU)')

    axs01[1].plot(autofluo_glucose_ramp_regime.time,
                  autofluo_glucose_ramp_regime.glucose_conc,
                 color = 'grey')
    for event_time in autofluo_glucose_ramp_regime.time:
        axs01[1].axvline(event_time, ymin = 0, ymax = 1,
                         color = 'lightgrey', linestyle = '--')
    axs01[1].set_title('Glucose concentration in microfluidics chamber',
                       size = 10)
    axs01[1].set_ylabel('Glucose concentration (g/L)')

    fig01.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor='none', \
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Time (min)')
    plt.show()
