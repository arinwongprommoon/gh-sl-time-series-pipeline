#!/usr/bin/env python3
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from . import rsetattr, rgetattr

class ROC:
    """
    Container for attributes relevant to ROC curve, plus methods for
    computing and plotting

    Attributes:
    -----------
    x = 1D numpy array,
        axis of false positive rates
    y = 1D numpy array,
        axis of true positive rates
    s = 1D numpy array,
        axis of distances from (0,1)
    bestqs = 1D numpy array,
        optimal q values
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.s = None
        self.bestqs = None

    def compute(self,
            Dataset,
            scoring_method,
            vec = np.power(10, np.linspace(-14, -8, 10))):
        """
        Computes axes for ROC curve and stores in attributes

        Parameters:
        -----------
        Dataset = a DataAttr objecct
        vec = vector of q-values to sweep over
        """
        x = []
        y = []
        for q in vec:
            Dataset.fdr_classify(scoring_method = scoring_method, q = q)
            Dataset.summarise_classification()
            Dataset.add_cm()
            Dataset.add_fpr()
            Dataset.add_tpr()

            x.append(Dataset.fpr)
            y.append(Dataset.tpr)
        self.x = np.array(x)
        self.y = np.array(y)

        self.s = np.sqrt(self.x**2 + (1 - self.y)**2)
        self.bestqs = vec[self.s == min(self.s)]

    def plot(self, Dataset):
        """
        Plots ROC curve from stored attributes, runs compute() if appropriate
        """
        if self.x is None:
            self.compute(Dataset = Dataset)

        fig, ax = plt.subplots(figsize = (3,3))
        ax.plot(self.x, self.y)
        ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle = ':')
        ax.set(title = 'ROC curve',
                xlabel = 'False positive rate',
                ylabel = 'True positive rate')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])

        return fig, ax

def round(n, k):
    """
    rounds 'n' up/down to the nearest 'k'
    use positive k to round up
    use negative k to round down
    """
    return n - n % k

def kymograph(list_CellAttr, cell_attr,
              order_by='cellid', normalise=False, mode='heatmap'):
    # This is probably the most poorly-written code in the entire pipeline.
    # It's impossible to understand and even more difficult to modify.
    """
    Draws kymograph

    Parameters:
    -----------
    list_CellAttr = a list of CellAttr objects
    cell_attr = string,
        CellAttr attribute to plot e.g. 'fluo' or 'fluo_processed'
    order_by_rank = string,
        CellAttr attribute to rank by.  Is cellid if not specified (i.e. not
        ranking by anything)
    normalise = boolean,
        specify whether to divide time series by range
    mode = string,
        'heatmap', 'heatmap_split', 'line'; line plots just the first 20
    """
    # how to order the time series
    order = np.argsort([rgetattr(cell, order_by) for cell in list_CellAttr])
    # length of time series
    # assumes all ts are same length; this holds even if some have NaNs
    l = len(rgetattr(list_CellAttr[0], cell_attr))

    # construct np array that contains the data
    if normalise:
        kg_list = [rgetattr(list_CellAttr[ii], cell_attr)[jj]/ \
                    np.ptp(rgetattr(list_CellAttr[ii], cell_attr))
                for ii in order
                for jj in range(l)]
    else:
        kg_list = [rgetattr(list_CellAttr[ii], cell_attr)[jj]
                for ii in order
                for jj in range(l)]
    kg_array = np.array(kg_list).reshape(len(list_CellAttr), l)

    # x ticks
    xtick_step = 60
    xtick_min = int(round(list_CellAttr[0].time.min(), (-1*xtick_step)))
    xtick_max = (int(round(list_CellAttr[0].time.max(), (1*xtick_step)))) \
            + xtick_step
    xticklabels = range(xtick_min, xtick_max, xtick_step)
    xticks = []
    for label in xticklabels:
        idx_pos = int(np.asscalar(np.where(list_CellAttr[0].time == label)[0]))
        xticks.append(idx_pos)

    if mode == 'heatmap':
        seaborn.set()
        fig, ax = plt.subplots()
        ax = seaborn.heatmap(kg_array, center = 0, cmap = "vlag", square=True)
        ax.set_ylabel('Cell')
        ax.set_xlabel('Time (min)')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    elif mode == 'heatmap_split':
        attr_list = [rgetattr(list_CellAttr[ii], order_by) \
                     for ii in range(len(list_CellAttr))]
        # number of unique values of the cell_attr, used to define number of
        # groups in kymograph
        ngroups = len(np.unique(attr_list))
        # list to define where to split kg_array
        groupcounts = list(pd.Series(attr_list).value_counts(sort=False))
        segments = np.cumsum(groupcounts)
        vmin = np.min(kg_array)
        vmax = np.max(kg_array)
        fig, axs = plt.subplots(nrows = ngroups, ncols = 1, sharex = True,
                               gridspec_kw = dict(height_ratios=groupcounts))
        for ii in range(ngroups):
            if ii == 0:
                axs[ii] = seaborn.heatmap(kg_array[0:segments[0]],
                                          center = 0, cmap = "vlag",
                                          ax = axs[ii], square = True,
                                          cbar = False,
                                          vmin = vmin, vmax = vmax,)
                axs[ii].set_ylabel('Cell')
                axs[ii].set_xlabel('Time (min)')
                axs[ii].set_xticks(xticks)
                axs[ii].set_xticklabels(xticklabels)

            else:
                axs[ii] = seaborn.heatmap(kg_array[segments[ii-1]:segments[ii]],
                                          center = 0, cmap = "vlag",
                                          ax = axs[ii], square = True,
                                          cbar = False,)
                axs[ii].set_ylabel('Cell')
                axs[ii].set_xlabel('Time (min)')
                axs[ii].set_xticks(xticks)
                axs[ii].set_xticklabels(xticklabels)
    elif mode == 'line':
        # just the first 20 because then matplotlib dies
        fig, axs = plt.subplots(nrows = 20, ncols = 1,
                                sharex = True, sharey = True,
                                gridspec_kw = {'hspace': 0})
        t = np.linspace(0, 2.5*(l-1), l)
        for ii in range(20):
            axs[ii].set_yticks([])
            axs[ii].plot(t, kg_array[ii])
        for ax in axs:
            ax.label_outer()

    #return fig, axs

def violin(list_CellAttr, group_by, feature_list = list(range(22))):
    # Analogue of TS_TopFeatures() in hctsa
    """
    Draws violin plots that represents normalised specified catch22 feature
    values for a list of cells, with subplots for each group specified by a
    given attribute.

    Parameters:
    -----------
    list_CellAttr = list of CellAttr objects,
        list of CellAttr objects from which to extract catch22 features
        (should be stored in CellAttr.hctsa_vec)
    group_by = string,
        CellAttr attribute to group the cells by
    feature_list = list of integers,
        list of indices (0 to 21) of catch22 features to draw violin plots of.
        Order matters -- order of indices will be reflected in plot
        Default: plots all features in order
    """
    # gets information about groups by specified attribute, and puts in in
    # appropriate variable type
    attr_list = [rgetattr(list_CellAttr[ii], group_by) \
                 for ii in range(len(list_CellAttr))]
    groups = np.unique(attr_list)
    ngroups = len(np.unique(attr_list))
    # prepare figure
    fig, axs = plt.subplots(nrows = len(groups), ncols = 1,
                           sharex = True, sharey = True)
    # each subplot = each group
    for group_index, group in enumerate(groups):
        # list of lists -- the latter is a list of normalised values for a 
        # catch22 feature within a cell group
        # structure to interface with matplotlib.axes.Axes.violinplot
        feature_values = [[cell.hctsa_vec[feature]
                           for cell in list_CellAttr
                           if rgetattr(cell, group_by) == group]
                          for feature in feature_list]
        # produces violin plot; positions and set_xticks specified so as to
        # produce evenly-spaced violins
        axs[group_index].violinplot(feature_values,
                             positions = list(range(len(feature_list))),
                             showmeans = True, showextrema = True,
                             showmedians = True, bw_method = 0.5)
        axs[group_index].set_xticks(list(range(len(feature_list))))
        # label with features, preserve the order in the parameters
        axs[group_index].set_xticklabels(feature_list)
        axs[group_index].set_ylabel('Normalised value')
    plt.xlabel('Feature')

def order_cells(list_CellAttr, get = 'cellid', by = 'rank'):
    """
    Returns list of some attr by another attr, default list of cellids
    by rank
    """
    s = sorted(list_CellAttr, key = lambda cell: rgetattr(cell, by))
    return np.array([rgetattr(cell, get) for cell in s])

def cell_cycle_durations(list_CellAttr):
    """
    Returns array of cell cycle durations from births of CellAttr objects
    """
    d = [np.diff(cell.births) for cell in list_CellAttr]
    d = np.concatenate(d).ravel()
    return d
