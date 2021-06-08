#!/usr/bin/env python3
import numpy as np
import scipy as sp
import pandas as pd
import itertools
import matplotlib
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
    cell.strain = 'FY4'

    cell.flavin.exposure = 60 * ((cell.position - 1)//3)
    cell.flavin.reading = cell.y
    cell.flavin.category = Flavin_dcategory[ii]

    cell.mCherry = pipeline.Fluo('mCherry')
    cell.mCherry.exposure = 0

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
        cell.strain = 'FY4'
    elif cell.position in [7,8,9,10]:
        cell.strain = 'Whi5-mCherry'

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
allcells = Flavin.cells + Whi5.cells# + Baum18.cells

for cell in allcells:
    # Fourier transform
    add_classicalAttr(cell, oversampling_factor = 1)
    #add_bglsAttr(cell)
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

allcells0 = Flavin.cells + Whi5.cells + Baum18.cells
# Creates catch22 data matrix
catch22_dataMatrix = np.array(
    [catch22.catch22_all(cell.flavin.reading_processed)['values']
     for cell in allcells0])
# Normalises catch22 data matrix
catch22_dataMatrix_norm = featext.tsman.TS_Normalize(catch22_dataMatrix)
# Stuffs normalised data as feature vectors in a new attribute for each cell
for ii, cell in enumerate(allcells0):
    cell.hctsa_vec = catch22_dataMatrix_norm[ii]

# Compute distance matrix
catch22_distanceMatrix = sklearn.metrics.pairwise.cosine_distances( \
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
for ii, cell in enumerate(allcells0):
    cell.flavin.cluster = allgraph_partition.membership[ii]


## FIGURE GENERATION
# representative time series
if False:
    fig01_listCellAttr = [Flavin.cells[154],
                         Whi5.cells[23],
                         Whi5.cells[65]]
    fig01, axs01 = plt.subplots(nrows = len(fig01_listCellAttr), ncols = 1,
                                sharex = True, sharey = False,
                                gridspec_kw = {'hspace': 0.5})
    for ii, cell in enumerate(fig01_listCellAttr):
        axs01[ii].plot(cell.time, cell.flavin.reading_processed)
        if cell.births.any():
            for birth in cell.births:
                axs01[ii].axvline(birth, ymin = 0, ymax = 1,
                                  color = 'r', linestyle = '--')
        axs01[ii].set_title(cell.strain + ' strain in ' + cell.medium.base + \
                       ', flavin LED ' + str(cell.flavin.exposure) + \
                       ' ms, mCherry LED ' + str(cell.mCherry.exposure) + ' ms', \
                           size = 10)
    fig01.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', \
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Time (min)')
    plt.ylabel('Flavin fluorescence (AU)')
    plt.show()

# flavin vs whi5
if True:
    cell = Whi5.cells[65]
    fig02, axs02 = plt.subplots(nrows = 2, ncols = 1,
                            sharex = True, sharey = False,
                            gridspec_kw = {'hspace': 0.2})
    axs02[0].plot(cell.time, cell.flavin.reading, color = 'b')
    if cell.births.any():
        for birth in cell.births:
            axs02[0].axvline(birth, ymin = 0, ymax = 1,
                          color = 'r', linestyle = '--')
    axs02[0].set_ylabel('Flavin fluorescence (AU)')
    axs02[1].plot(cell.time, cell.mCherry.reading, color = 'g')
    if cell.births.any():
        for birth in cell.births:
            axs02[1].axvline(birth, ymin = 0, ymax = 1,
                          color = 'r', linestyle = '--')
    fig02.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', \
                top=False, bottom=False, left=False, right=False)
    axs02[1].set_ylabel('Whi5 localisation (AU)')
    plt.xlabel('Time (min)')
    plt.show()

# power spectra of representative traces
if False:
#    fig03_listCellAttr = [Flavin.cells[72], Flavin.cells[154], Flavin.cells[233],
#                         Whi5.cells[42], Whi5.cells[41], Whi5.cells[23],
#                         Whi5.cells[11], Whi5.cells[67], Whi5.cells[65]]
    fig03_listCellAttr = [Flavin.cells[154],
                         Whi5.cells[23],
                         Whi5.cells[65]]
    fig03, axs03 = plt.subplots(nrows = len(fig03_listCellAttr), ncols = 1,
                                sharex = True, sharey = False,
                                gridspec_kw = {'hspace': 0.5})
    for ii, cell in enumerate(fig03_listCellAttr):
        axs03[ii].plot(cell.flavin.classical.freqs[0:45],
                       cell.flavin.classical.power[0:45])
        axs03[ii].set_title(cell.strain + ' strain in ' + cell.medium.base + \
                       ', flavin LED ' + str(cell.flavin.exposure) + \
                       ' ms, mCherry LED ' + str(cell.mCherry.exposure) + ' ms', \
                           size = 10)
    fig03.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', \
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Frequency ($min^{-1}$)')
    plt.ylabel('Power (dimensionless)')
    plt.show()

# ugly ymc vs cdc scatter plot
if False:
    for cell in allcells:
        cell.flavin.classical.add_pd()
    sm_ymc_pds = [cell.flavin.classical.pd for cell in Flavin.cells]
    sc_ymc_pds = [cell.flavin.classical.pd for cell in Whi5.cells
                  if cell.strain == 'FY4']
    whi5_ymc_pds = [cell.flavin.classical.pd for cell in Whi5.cells
                    if cell.strain == 'Whi5-mCherry']
    sm_cdc_pds = [np.mean(np.diff(cell.births)) for cell in Flavin.cells]
    sc_cdc_pds = [np.mean(np.diff(cell.births)) for cell in Whi5.cells
                 if cell.strain == 'FY4']
    whi5_cdc_pds = [np.mean(np.diff(cell.births)) for cell in Whi5.cells
                    if cell.strain == 'Whi5-mCherry']
    colourmap = np.array(matplotlib.cm.Dark2.colors)
    fig04, ax04 = plt.subplots(figsize = (6,6))
    ax04.scatter(sm_cdc_pds, sm_ymc_pds,
                 color = colourmap[0], marker = 'o', alpha = 0.5)
    ax04.scatter(sc_cdc_pds, sc_ymc_pds,
                 color = colourmap[1], marker = 'o', alpha = 0.5)
    ax04.scatter(whi5_cdc_pds, whi5_ymc_pds,
                 color = colourmap[2], marker = 'o', alpha = 0.5)
    ax04.legend(['FY4 in SM', 'FY4 in SC', 'WHI5::mCherry in SC'])
    ax04.plot(np.linspace(0,800,801), np.linspace(0,800,801),
             c = 'k', linestyle = '--')
    plt.xlabel('Mean cell division cycle duration of cell (min)')
    plt.ylabel('Estimated YMC period of cell (min)')
    plt.show()

# cdc histogram
if False:
    histdata = [pipeline.vis.cell_cycle_durations(Flavin.cells),
                pipeline.vis.cell_cycle_durations([cell for cell in Whi5.cells
                                                   if cell.strain == 'FY4']),
                pipeline.vis.cell_cycle_durations([cell for cell in Whi5.cells
                                                   if cell.strain == 'Whi5-mCherry'])]
    colourmap = np.array(matplotlib.cm.Dark2.colors)
    fig05, ax05 = plt.subplots()
    ax05.hist(histdata, bins = 60,
              histtype = 'step',
              color = [colourmap[0], colourmap[1], colourmap[2]])
        # colours to match previous plot
    ax05.legend(['WHI5::mCherry in SC', 'FY4 in SC', 'FY4 in SM'])
    plt.xlabel('Cell division cycle duration (min)')
    plt.ylabel('Number of cell division cycles')
    plt.show()

# exposure histogram
if False:
    sc_histdata = \
        [pipeline.vis.cell_cycle_durations([cell for cell in Flavin.cells
                                            if cell.flavin.exposure == exposure])
         for exposure in [60, 120, 180]]
    sm_histdata = \
        [pipeline.vis.cell_cycle_durations([cell for cell in Whi5.cells
                                            if cell.flavin.exposure == exposure])
         for exposure in [60, 120]]
    colourmap = np.array(matplotlib.cm.Set1.colors)
    fig06, axs06 = plt.subplots(nrows = 2, ncols = 1, sharex = True)
    axs06[0].hist(sc_histdata, bins = 60,
              histtype = 'step',
              color = [colourmap[0], colourmap[1], colourmap[2]])
        # colours to match previous plot
    axs06[0].legend(['60 ms', '120 ms', '180 ms'])
    axs06[0].set_title('Cells in SM')
    axs06[1].hist(sm_histdata, bins = 60,
              histtype = 'step',
              color = [colourmap[1], colourmap[2]])
    axs06[1].legend(['60 ms', '120 ms'])
    axs06[1].set_title('Cells in SC')
    fig06.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', \
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Cell division cycle duration (min)')
    plt.ylabel('Number of cell division cycles')
    plt.show()

# best and worst ranked cells in Flavin+Whi5 set
if False:
    FlavinWhi5 = pipeline.DatasetAttr(Flavin.cells + Whi5.cells)
    # redefine cellids
    for ii, cell in enumerate(FlavinWhi5.cells):
        cell.cellid = ii
    # recompute ranks
    FlavinWhi5.fdr_classify(q = 1e-8, scoring_method = 'glynn06')
    # Xg values will already have been included -- this is worst to best
    ranking = pipeline.vis.order_cells(FlavinWhi5.cells,
                                       by = 'flavin.glynn06.score')

    fig07, axs07 = plt.subplots(nrows = 2, ncols = 2, sharex = True)
    worst5 = ranking[0:2]
    best5 = ranking[-3:-1]
    for ii, cellid in enumerate(np.concatenate((best5,worst5))):
        cell = FlavinWhi5.cells[cellid]
        axs07[ii % 2, ii // 2].plot(cell.time, cell.flavin.reading_processed)
        if cell.births.any():
            for birth in cell.births:
                axs07[ii % 2, ii // 2].axvline(birth, ymin = 0, ymax = 1,
                                  color = 'r', linestyle = '--')
        axs07[ii % 2, ii // 2].set_title(cell.strain + ' strain in ' + cell.medium.base + \
                       ', flavin LED ' + str(cell.flavin.exposure) + \
                       ' ms, mCherry LED ' + str(cell.mCherry.exposure) + ' ms', \
                           size = 10)
    fig07.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', \
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Time (min)')
    plt.ylabel('Flavin pipelinerescence (AU)')
    plt.show()

# just plots user-defined sets of time series in 2 columns
if False:
    fig07, axs07 = plt.subplots(nrows = 5, ncols = 2, sharex = True)
    worst10 = [202, 71, 104, 222, 14]#, 127, 176, 25, 97, 61]
    best10 = [36, 72, 154, 107, 218]#, 67, 53, 114, 188, 4]
    for ii, cellid in enumerate(np.concatenate((best10,worst10))):
        cell = Flavin.cells[cellid]
        axs07[ii % 5, ii // 5].plot(cell.time, cell.flavin.reading_processed)
        #axs07[ii % 10, ii // 10].set_title('Cell ' + str(cell.cellid), size = 10)
    fig07.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', \
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Time (min)')
    plt.ylabel('Flavin pipelinerescence (AU)')
    plt.show()

# as above but power spectra
if False:
    FlavinWhi5 = pipeline.DatasetAttr(Flavin.cells + Whi5.cells)
    # redefine cellids
    for ii, cell in enumerate(FlavinWhi5.cells):
        cell.cellid = ii
    # recompute ranks
    FlavinWhi5.fdr_classify(q = 1e-8, scoring_method = 'glynn06')
    # Xg values will already have been included -- this is worst to best
    ranking = pipeline.vis.order_cells(FlavinWhi5.cells,
                                       by = 'flavin.glynn06.score')

    fig07, axs07 = plt.subplots(nrows = 2, ncols = 2,
                                sharex = True, sharey = True)
    worst5 = ranking[0:2]
    best5 = ranking[-3:-1]
    for ii, cellid in enumerate(np.concatenate((best5,worst5))):
        cell = FlavinWhi5.cells[cellid]
        axs07[ii % 2, ii // 2].plot(cell.flavin.classical.freqs[0:45],
                                    cell.flavin.classical.power[0:45])
        axs07[ii % 2, ii // 2].set_title(cell.strain + ' strain in ' + cell.medium.base + \
                       ', flavin LED ' + str(cell.flavin.exposure) + \
                       ' ms, mCherry LED ' + str(cell.mCherry.exposure) + ' ms', \
                           size = 10)
    fig07.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', \
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Frequency (min)')
    plt.ylabel('Power (dimensionless)')
    plt.show()

# as above but power spectra and time series as inset
if False:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    FlavinWhi5 = pipeline.DatasetAttr(Flavin.cells + Whi5.cells)
    # redefine cellids
    for ii, cell in enumerate(FlavinWhi5.cells):
        cell.cellid = ii
    # recompute ranks
    FlavinWhi5.fdr_classify(q = 1e-8, scoring_method = 'glynn06')
    # Xg values will already have been included -- this is worst to best
    ranking = pipeline.vis.order_cells(FlavinWhi5.cells,
                                       by = 'flavin.glynn06.score')

    fig07, axs07 = plt.subplots(nrows = 2, ncols = 2,
                                sharex = True, sharey = True)
    worst5 = ranking[0:2]
    best5 = ranking[-3:-1]
    for ii, cellid in enumerate(np.concatenate((best5,worst5))):
        cell = FlavinWhi5.cells[cellid]
        axs07[ii % 2, ii // 2].plot(cell.flavin.classical.freqs[0:45],
                                    cell.flavin.classical.power[0:45])
        axs07[ii % 2, ii // 2].set_title(cell.strain + ' strain in ' + cell.medium.base + \
                       ', flavin LED ' + str(cell.flavin.exposure) + \
                       ' ms, mCherry LED ' + str(cell.mCherry.exposure) + ' ms', \
                           size = 10)
        axs_i = inset_axes(axs07[ii % 2, ii // 2],
                                width = "67%",
                                height = 2.0)
        plt.plot(cell.time, cell.flavin.reading_processed)
        if cell.births.any():
            for birth in cell.births:
                plt.axvline(birth, ymin = 0, ymax = 1,
                                  color = 'r', linestyle = '--')
        plt.xlabel('Time (min)')
        plt.ylabel('Flavin fluorescence (AU)')
    fig07.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', \
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel('Frequency (min)')
    plt.ylabel('Power (dimensionless)')
    plt.show()

# distance matrix
if False:
    fig08, axs08 = plt.subplots()
    # 'cheating' to draw boundaries between datasets
    catch22_distanceMatrix[243,:] = 1
    catch22_distanceMatrix[:,243] = 1
    catch22_distanceMatrix[312,:] = 1
    catch22_distanceMatrix[:,312] = 1
    plt.imshow(catch22_distanceMatrix)
    plt.colorbar()
    plt.show()

# demostrating validity of top catch22 features in discriminating sc vs sm
# i.e. firstMin_acf (first minimum of autocorrelation function) and
# SP_Summaries_welch_rect_centroid (centroid of power spectrum)

if False:
    from statsmodels.graphics.tsaplots import plot_acf
    sm = Flavin.cells[72]
    sc = Whi5.cells[42]

    # first minimum of autocorr
    def firstMin_acf(x):
        # makes blank figure so that plt.acorr dumps it there instead of
        # fucking up with the main figure
        plt.figure()
        lags, c, line, b = plt.acorr(x, maxlags = None)
        # chops lag and autocorr vector in half
        lags2 = lags[len(lags)//2:len(lags)]
        c2 = c[len(c)//2:len(c)]
        # first local minimum of autocorr vector
        peaks, properties = sp.signal.find_peaks(-c2)
        # returns time of first minimum
        return lags2[peaks[0]]

    # centroid of power spectrum
    def ps_centroid(freqs, power):
        power_cumulative = np.cumsum(power)/np.max(np.cumsum(power))
        over_midpoint = np.where(power_cumulative >= 0.5)[0]
        return freqs[over_midpoint[0]]

    # plotting
    fig09, axs09 = plt.subplots(nrows = 3, ncols = 2)

    def plot_row_catch22validity(cell, celltype, col): # saves duplicating myself
        # time series
        axs09[0,col].plot(cell.time, cell.flavin.reading_processed)
        axs09[0,col].set_xlabel('Time (min)')
        axs09[0,col].set_ylabel('Flavin fluorescence (AU)')
        axs09[0,col].set_title('Time series for selected ' + celltype + ' cell')

        # autocorrelation function
        fm = firstMin_acf(cell.flavin.reading_processed)
        axs09[1,col].acorr(cell.flavin.reading_processed, maxlags = None)
        axs09[1,col].set_xlim(left = 0)
        axs09[1,col].axvline(fm,
                           ymin = 0, ymax = 1, color = 'r', linestyle = '--')
        axs09[1,col].text(fm+10, 0.8,
                        'First minimum = ' + str(fm), color = 'r')
        axs09[1,col].set_xlabel('Lag time (min)')
        axs09[1,col].set_ylabel('Autocorrelation')
        axs09[1,col].set_title('Autocorrelation function (' + celltype + ')')

        # fourier spectrum + centroid
        centroid = ps_centroid(cell.flavin.classical.freqs,
                               cell.flavin.classical.power)
        axs09[2,col].plot(cell.flavin.classical.freqs,
                          cell.flavin.classical.power)
        axs09[2,col].axvline(centroid,
                           ymin = 0, ymax = 1, color = 'r', linestyle = '--')
        axs09[2,col].text(centroid+0.001, np.max(cell.flavin.classical.power)/2,
                        'Centroid = ' + '%.4f' % centroid, color = 'r')
        axs09[2,col].set_xlabel('Frequency ($min^{-1}$)')
        axs09[2,col].set_ylabel('Power (dimensionless)')
        axs09[2,col].set_title('Fourier power spectrum (' + celltype + ')')

    plot_row_catch22validity(sm, 'SM', 0)
    plot_row_catch22validity(sc, 'SC', 1)
    
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)

    plt.show()
