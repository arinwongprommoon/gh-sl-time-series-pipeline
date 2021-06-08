#!/usr/bin/env python3
import os

import numpy as np
import scipy as sp
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import sklearn.metrics
import igraph as ig

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

os.chdir('./data/arin')

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

def add_autoregAttr(cell):
    """
    Computes autoregressive model-based periodogram and adds PdgramAttr
    attributes
    """
    cell.flavin.autoreg = pipeline.PdgramAttr()
    cell.flavin.autoreg.label = \
            'Autogressive Model-Based Periodogram (Jia & Grima, 2020)'
    cell.flavin.autoreg.power_label = 'Power'
    freq_npoints = 1000
    cell.flavin.autoreg.freqs, cell.flavin.autoreg.power = \
            pipeline.periodogram.autoreg(cell.time,
                                         cell.flavin.reading_processed,
                                         freq_npoints)

# Import information for flavin exposure experiment from files
Flavin_rawdata = pipeline.dataimport.import_timeseries( \
        'Flavinexpostest3_ffcorr_small.csv')
Flavin_dcategory = pipeline.dataimport.import_categories( \
        'Flavinexpostest3_ffcorr_small_OscillationEvals.txt')
Flavin_births = pipeline.dataimport.import_births( \
        'Flavinexpostest3_ffcorr_small_births.csv')

# Import information for SC+Whi5-mCherry experiment from files
Whi5_rawdata = pipeline.dataimport.import_timeseries( \
        'FlavinWhi5_3_ffcorr_small.csv')
Whi5_rawdata_mCherry_unsliced = pipeline.dataimport.import_timeseries( \
        'FlavinWhi5_3_ffcorr_small_mCherry.csv')
# restrict to cells with flavin readings
idx_both = list(set(Whi5_rawdata.cellID) & set(Whi5_rawdata_mCherry_unsliced.cellID))
Whi5_rawdata_mCherry = \
        Whi5_rawdata_mCherry_unsliced.loc[ \
            Whi5_rawdata_mCherry_unsliced.cellID.isin(idx_both)]
Whi5_dcategory = pipeline.dataimport.import_categories( \
        'FlavinWhi5_3_ffcorr_small_OscillationEvals.txt')
#Whi5_dcategory = [3] * len(Whi5_rawdata)
Whi5_births = pipeline.dataimport.import_births( \
        'FlavinWhi5_3_ffcorr_small_births.csv')

# Import information from Baumgartner et al. (2018)
Baum18_rawdata = pd.read_csv('baumgartner2018/Baumgartner2018_AllNormData.csv',
                             header = 0)
# Counts number of columns called 'time_*' and 'flavin_*'
Baum18_time_ncols = len([s for s in Baum18_rawdata.columns.values if "time_" in s])
Baum18_flavin_ncols = len([s for s in Baum18_rawdata.columns.values if "flavin_" in s])
# Loads matrix that stores manually-defined oscillation categories
Baum18_dcategory0 = \
        pd.read_csv('baumgartner2018/Baumgartner2018_OscillationEvals.csv',
                    sep = ',', header = None)
# Reshapes it and removes zeros
Baum18_dcategory = Baum18_dcategory0.values.flatten()
Baum18_dcategory = Baum18_dcategory[np.nonzero(Baum18_dcategory)]

# Arranges information into DatasetAttr objects
# Reverted to this method rather than using CellAttr_from_datasets because now
# I'm working with different datasets that are formatted differently
Flavin_data = pipeline.dataimport.CellAttr_from_datasets( \
        timeseries_df = Flavin_rawdata,
        categories_array = Flavin_dcategory,
        births_df = Flavin_births)
Flavin = pipeline.DatasetAttr(Flavin_data)
# Add some labels
for ii, cell in enumerate(Flavin.cells):
    cell.source = 'flavin'
    cell.medium.base = 'SM'
    cell.medium.nutrients = {'glucose': 2}
    cell.strain = 'YST365'

    cell.flavin.exposure = 60 * ((cell.position - 1)//3)
    cell.flavin.reading = cell.y
    cell.flavin.category = Flavin_dcategory[ii]

Whi5_data = pipeline.dataimport.CellAttr_from_datasets( \
        timeseries_df = Whi5_rawdata,
        categories_array = Whi5_dcategory,
        births_df = Whi5_births)
Whi5 = pipeline.DatasetAttr(Whi5_data)
# dummy -- will be better when I re-structure things... am just re-using a 
# function for quick-and-dirty purposes
Whi5_data_mCherry = pipeline.dataimport.CellAttr_from_datasets( \
        timeseries_df = Whi5_rawdata_mCherry,
        categories_array = Whi5_dcategory,
        births_df = Whi5_births)
Whi5_mCherry = pipeline.DatasetAttr(Whi5_data_mCherry)
Whi5_mCherry_MATLABids = [cell.MATLABid for cell in Whi5_mCherry.cells]
# Add some labels
for ii, cell in enumerate(Whi5.cells):
    cell.source = 'whi5'
    cell.medium.base = 'SC'
    cell.medium.nutrients = {'glucose': 2}

    # if I had formatted my CSV to include these values, I wouldn't have to do
    # this admittedly VERY bad way of assigning attribute values
    if cell.position in [1,2,3,4]:
        cell.strain = 'YST365'
    elif cell.position in [7,8,9,10]:
        cell.strain = 'YST556'

    cell.flavin = pipeline.Fluo('flavin')
    if cell.position in [1,2,7,8]:
        cell.flavin.exposure = 120
    elif cell.position in [3,4,9,10]:
        cell.flavin.exposure = 60
    cell.flavin.reading = cell.y
    cell.flavin.category = Whi5_dcategory[ii]

    cell.mCherry = pipeline.Fluo('mCherry')
    if cell.position in [1,3,7,9]:
        cell.mCherry.exposure = 100
    elif cell.position in [2,4,8,10]:
        cell.mCherry.exposure = 0
    # loads in reading, cross-referencing by MATLABid.  This is awful, I know.
    if cell.MATLABid in Whi5_mCherry_MATLABids:
        cell.mCherry.reading = \
            Whi5_mCherry.cells[Whi5_mCherry_MATLABids.index(cell.MATLABid)].y

# (shamelessly copied from my older code written specifically for this dataset
# BUT SOMEHOW IT FUCKING WORKS)
Baum18 = pipeline.DatasetAttr([])
for ii in range(len(Baum18_rawdata)):
    dtime = np.array(Baum18_rawdata.iloc[ii][1:(Baum18_time_ncols+1)])
    dtime = dtime[~np.isnan(dtime)]
    dflavin = np.array(Baum18_rawdata.iloc[ii][(Baum18_time_ncols+1):\
                         (Baum18_time_ncols+Baum18_flavin_ncols+1)])
    dflavin = dflavin[~np.isnan(dflavin)]

    Baum18.cells.append(pipeline.CellAttr(cellid = ii))

    Baum18.cells[ii].cellid = int(Baum18_rawdata.iloc[ii][0])
    # Baum18.cells[ii].medium, strain...

    Baum18.cells[ii].time = dtime
    Baum18.cells[ii].flavin = pipeline.Fluo('flavin')
    Baum18.cells[ii].flavin.reading = dflavin
    Baum18.cells[ii].flavin.reading_processed = dflavin
    Baum18.cells[ii].flavin.category = Baum18_dcategory[ii]

    Baum18.cells[ii].source = 'baum18'

# Does things that are independent between cells
for cell in itertools.chain(Flavin.cells, Whi5.cells):
    # Filters time series
    cell.flavin.reading_processed = \
            pipeline.tsman.stdfilter(cell.flavin.reading, Fs = 1/2.5)

pipeline.tsman.population_detrend(Flavin.cells, 'flavin.reading_processed')
pipeline.tsman.population_detrend(Whi5.cells, 'flavin.reading_processed')

# Master list for iteration
allcells = Flavin.cells #+ Whi5.cells + Baum18.cells

for cell in allcells:
    # Fourier transform
    add_classicalAttr(cell, oversampling_factor = 1)
    #add_bglsAttr(cell)
    add_autoregAttr(cell)
    # Scores using Glynn et al. 2006
    cell.flavin.glynn06 = pipeline.score.glynn06(cell)

# dummy q; I just want ranks
Flavin.fdr_classify(q = 1e-8, scoring_method = 'glynn06')
Whi5.fdr_classify(q = 1e-8, scoring_method = 'glynn06')

# Does things at the level of dataset

# ROC curve
#myROC = pipeline.vis.ROC()
#myROC.compute(d)
#myROC.plot(d)

# uses best q from ROC
#bestq = myROC.bestqs[0]

# various ML stuff in usual routine
#d.fdr_classify(q = bestq)
#d.summarise_classification()
#d.add_cm()
#d.add_fpr()
#d.add_tpr()
#d.add_fdr()
#d.add_accuracy()

# THIS BELONGS IN A DIFFERENT SCRIPT BUT WILL ORGANISE IT LATER
# or maybe jupyter notebook

if False:
    # Creates catch22 data matrix
    catch22_dataMatrix = np.array(
        [catch22.catch22_all(cell.flavin.reading_processed)['values']
         for cell in allcells])
    # Normalises catch22 data matrix
    catch22_dataMatrix_norm = featext.tsman.TS_Normalize(catch22_dataMatrix)
    # Stuffs normalised data as feature vectors in a new attribute for each cell
    for ii, cell in enumerate(allcells):
        cell.hctsa_vec = catch22_dataMatrix_norm[ii]

    # Compute distance matrix
    catch22_distanceMatrix = sklearn.metrics.pairwise.euclidean_distances( \
        catch22_dataMatrix_norm)

    # Prune distance matrix
    catch22_distanceMatrix_pruned = \
        featext.graph.prune(distanceMatrix = catch22_distanceMatrix,
                                   neighbours = 10)
    # Creates graph
    allgraph = ig.Graph.Weighted_Adjacency(catch22_distanceMatrix_pruned.tolist(),
                                           mode = 'undirected')
    # Partitions graph.  Note that the resolution parameter is defined differently
    # from Mucha et al. (2010)
    allgraph_partition = \
        leidenalg.find_partition(allgraph, leidenalg.ModularityVertexPartition)

    # Saves clusters into new attribute in CellAttr
    for ii, cell in enumerate(allcells):
        cell.flavin.cluster = allgraph_partition.membership[ii]
