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
dropTPs_num = 0
# sampling interval (min)
sampling_interval = 2.5

def dropTPs(df, dropTPs_num):
    df.drop(df.columns[list(range(3, 3+dropTPs_num))], axis = 1, inplace = True)

# Import information from glucose shift experiment into dataframes
# (flavin_whi5_glucose_limitation_hard_03, 2020-12-10)

ChannelFlavin_rawdata = pipeline.dataimport.import_timeseries( \
        'Flavin-murray-hack_Flavin.csv')
# drops first couple time points because of imaging errors
dropTPs(ChannelFlavin_rawdata, dropTPs_num)
# dummy data for oscillation category -- not used for now
ChannelFlavin_dcategory = [3] * len(ChannelFlavin_rawdata)

Births = pipeline.dataimport.import_births( \
        'Flavin-murray-hack_births.csv')


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
GlucoseLimitation = \
        pipeline.DatasetAttr(GlucoseLimitation_ChannelFlavin_data)
# Add some labels
for ii, cell in enumerate(GlucoseLimitation.cells):
    cell.source = 'flavin_whi5_glucose_limitation_hard_03'
    cell.medium.base = 'SC'
    # NOTE TO SELF: future version of code should be able to chop up timeseries
    # according to different nutrient regimes
    cell.medium.nutrients = {'glucose': 0}

    # if I had formatted my CSV to include these values, I wouldn't have to do
    # this FUCKING AWFUL of assigning attribute values

    # Strains
    if cell.position in [1,2,3,4,5,6]:
        cell.strain = 'CEN.PK Mat A'
    elif cell.position in [7,8,9,10,11,12]:
        cell.strain = 'FY4'
    elif cell.position in [13,14,15,16,17,18]:
        cell.strain = 'CEN.PK Mat alpha'
    # Flavin channel
    cell.flavin = pipeline.Fluo('flavin')
    cell.flavin.exposure = 30
    cell.flavin.reading = cell.y
    cell.flavin.category = ChannelFlavin_dcategory[ii]

# Process data
for cell in GlucoseLimitation.cells:
    # apply notch filter
    cell.flavin.reading_processed = pipeline.tsman.notch_filter( \
            cell.flavin.reading, freq = 0.16, fs = 1/sampling_interval)


# Plotting

# Grouping the cells to make life easier
FY4Cells = \
        [cell for cell in GlucoseLimitation.cells if cell.strain == 'FY4']
CENPKMatACells = \
        [cell for cell in GlucoseLimitation.cells if cell.strain == 'CEN.PK Mat A']
CENPKMatalphaCells = \
        [cell for cell in GlucoseLimitation.cells if cell.strain == 'CEN.PK Mat alpha']

# Kymographs
pipeline.vis.kymograph(FY4Cells, cell_attr = 'flavin.reading_processed')
pipeline.vis.kymograph(CENPKMatACells, cell_attr = 'flavin.reading_processed')
pipeline.vis.kymograph(CENPKMatalphaCells, cell_attr = 'flavin.reading_processed')

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
    MeanCell.births = np.array([360, 840, 1080]) - \
            dropTPs_num * sampling_interval
    return MeanCell

FY4MeanCell = \
        mean_cell(FY4Cells, \
                  list_attr = ['flavin.reading_processed'])
FY4MeanCell.plot_ts(y_attr = 'flavin.reading_processed', births = True)
CENPKMatAMeanCell = \
        mean_cell(CENPKMatACells, \
                  list_attr = ['flavin.reading_processed'])
CENPKMatAMeanCell.plot_ts(y_attr = 'flavin.reading_processed', births = True)
CENPKMatalphaMeanCell = \
        mean_cell(CENPKMatalphaCells, \
                  list_attr = ['flavin.reading_processed'])
CENPKMatalphaMeanCell.plot_ts(y_attr = 'flavin.reading_processed', births = True)

# Example cells
FY4Cells[0].plot_ts(y_attr = 'flavin.reading_processed', births = False)
CENPKMatACells[0].plot_ts(y_attr = 'flavin.reading_processed', births = False)
CENPKMatalphaCells[0].plot_ts(y_attr = 'flavin.reading_processed', births = False)
