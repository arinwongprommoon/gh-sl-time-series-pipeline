#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools

__all__ = ["dataimport",
           "dataexport",
           "score",
           "periodogram",
           "tsman",
           "vis",
           "ar_grima2020"]

# Makes setting and getting attributes recursive
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class PdgramAttr:
    """
    Container for axis attributes for a given type of periodogram.

    Attributes:
    -----------
    label = string,
        name of type of periodogram, intended to be used as the plot title

    freqs = 1D array-like,
        frequency axis
    freqs_label = string,
        name of quantity represented by the frequency axis
    freqs_rec_label = string,
        name of quantity represented by the reciprocal of the frequency axis
    pd = float,
        most probable period

    power = 1D array-like,
        power axis
    power_label = string,
        name of quantity represented by the power axis
    """
    def __init__(self):
        self.label = 'Periodogram (spectrum)'

        self.freqs = None
        self.freqs_label = 'Frequency ($min^{-1}$)'
        self.freqs_rec_label = 'Period (min)'
        self.pd = None

        self.power = None
        self.power_label = 'Power (dimensionless)'

    def add_pd(self):
        self.pd = (1/self.freqs[self.power == max(self.power)])[0]

class ScoreAttr:
    """
    Container for cell's score for how good the time series is oscillating
    """
    def __init__(self):
        self.label = 'Xg'
        self.score = None

        class EmptyClass:
            def __init__(self):
                pass

        self.misc = EmptyClass() # exists solely for sfit, this is 100% NOT a
                                 # good idea

class Medium:
    """
    Container for information about nutrient medium used in experiment
    """
    def __init__(self):
        # SC, SM, etc
        self.base = None
        # dictionary: intention is keys are names of nutrients,
        # values are concentrations in percent w/v
        self.nutrients = {}

class Fluo:
    """
    Attributes for a particular fluorescence channel
    """
    def __init__(self, fluorophore):
        # name
        self.fluorophore = None
        # exposure time in ms
        self.exposure = None

        # raw fluorescence reading
        self.reading = None
        # space for processed
        self.reading_processed = None

        self.category = None
        self.classical = PdgramAttr()
        self.glynn06 = ScoreAttr()
        self.wichert03 = ScoreAttr()

    def plot_ps(self, pdgram = None, pd = False):
        """
        Generic power-spectrum plotting function.
        Axes taken from specifications in self.classical by default,
        this can be set to something else if another PdgramAttr attribute
        is present.

        Parameters:
        -----------
        pd = boolean
            specifies whether to plot the frequency axis as period

        Return:
        -------
        matplotlib.pyplot figure.Figure and axes.Axes objects
        """
        if pdgram is None:
            pdgram = self.classical
        else:
            pdgram = rgetattr(self, pdgram)

        fig, ax = plt.subplots()
        if pd:
            ax.plot(1/pdgram.freqs, pdgram.power)
            ax.set(title = pdgram.label,
                    xlabel = pdgram.freqs_rec_label,
                    ylabel = pdgram.power_label)
        else:
            ax.plot(pdgram.freqs, pdgram.power)
            ax.set(title = pdgram.label,
                    xlabel = pdgram.freqs_label,
                    ylabel = pdgram.power_label)

        return fig, ax

class CellAttr:
    """
    Cell-specific attributes and quantities derived from time series
list_CellAttr
    Attributes:
    -----------
    cellid = int,
    MATLABid = int,
        id corresponding to the index of cell in cExperiment for cross-checking
    position = int,
        position in device
    distfromcentre = float,
        distance cell is from centre of aperture in px

    strain = string,
        name of yeast cell strain
    medium =

    time = 1D array,
        time axis
    fluo = 1D array,
        axis for time series measurement of fluorescence

    category = int,
        manually-defined category of oscillation:
            1 - oscillating
            2 - unsure
            3 - non-oscillating
    births = 1D array,
        birth times in minutes

    rank = int,
        ranking of cell after scoring, should start from 1
    classification = boolean,
        classification of whether cell is oscillating or not according to
        algorithm

    This class also allows adding periodograms as PdgramAttr objects and scores
    as ScoreAttr objects.
    """
    def __init__(self, cellid):
        self.cellid = cellid
        self.MATLABid = None
        self.position = None
        self.distfromcentre = None

        self.strain = None
        self.medium = Medium()

        self.time = None
        self.y = None
        self.flavin = Fluo('flavin')

    def plot_ts(self, y_attr = None, births = True):
        """
        Plot time series (time axis taken from self.time)

        Parameters:
        -----------
        y_attr = string,
            attribute for vertical axis value
        processed = boolean,
            specified whether to use processed data (reading_processed vs
            reading) if available
        births = boolean
            specifies whether to plot births from self.births, default is true

        Return:
        -------
        matplotlib.pyplot figure.Figure and axes.Axes objects
        """
        if y_attr is None:
            y_attr = 'flavin.reading_processed'

        y = rgetattr(self, y_attr)

        fig, ax = plt.subplots()
        ax.plot(self.time, y)
        #ax.set_ylim((-0.1,0.1))
        if births:
            if self.births.any():
                for birth in self.births:
                    ax.axvline(birth, ymin = 0, ymax = 1,
                               color = 'r', linestyle = '--')
        else:
            pass
        ax.set(title = 'Autofluorescence of cell %d over time' % self.cellid,
                xlabel = 'Time (min)',
                ylabel = 'Autofluorescence (AU)')

        return fig, ax


class DatasetAttr:
    """
    Container for datasets of cells and population-level ML-related methods

    Attributes:
    -----------
    cells = list of CellAttr objects,
        the population of cells in a 'dataset'
    resultsummary = pandas dataframe,
        crosstab between manually-defined and algorithm-defined oscillation
        categories
    cm = 2 x 2 numpy array,
        confusion matrix
    fpr = float,
        false positive rate
    tpr = float,
        true positive rate
    accuracy = float
    """
    def __init__(self, list_CellAttr):
        self.cells = list_CellAttr
        self.resultsummary = None
        self.cm = None
        self.fpr = None
        self.tpr = None
        self.accuracy = None

    def fdr_classify(self,
                     scoring_method,
                     q):
        # currently can only do glynn06 or wichert03 and no structure to allow
        # for the derivatives of the fisher's g-statistic test as described by
        # mcsweeney, 2006.  ideally i'd like the functions in pipeline.score
        # to return a function that describes the function to be used for
        # classification to make such a structure possible.  will do this when
        # the need arises
        """
        Classifies cells by oscillation according to a specified q (FDR)

        scoring_method:
            'glynn06' - according to Glynn et al. (2006)
                        (pipeline.score.glynn06)
            'wichert03' - according to Wichert et al. (2003)
                          (pipeline.score.wichert03)
        """
        if scoring_method == 'wichert03':
            rank_array = np.array([
                [el.cellid for el in self.cells],
                [el.flavin.wichert03.score for el in self.cells]
                ])
            # sorts by score, in this case p-value
            rank_array = rank_array[:, rank_array[1].argsort()]
            # classification
            classification = np.array([rank_array[1,k] <= (k+1)*q/len(self.cells)
                for k in range(len(self.cells))])

        elif scoring_method == 'glynn06':
            # finds M for each time series
            M = []
            for cell in self.cells:
                # assumes evenly-spaced time series
                sampling_pd = cell.time[1] - cell.time[0]
                l_ts = cell.time[-1] - cell.time[0] # duration of ts
                f_lb = 1/l_ts # lower end of freq
                f_ub = 0.5 * (1/sampling_pd) # upper end of freq - Nyquist limit
                M.append(f_ub * l_ts) # VanderPlas (2018)
            M = np.array(M)

            rank_array = np.array([
                [el.cellid for el in self.cells],
                [el.flavin.glynn06.score for el in self.cells]
                ])
            # sorts by score, in this case -Xg (i.e. sorts in DESCENDING order
            # of Xg rather than ASCENDING order because high Xg = more likely
            # oscillating
            rank_array = rank_array[:, (-rank_array[1]).argsort()]
            # classification
            classification = np.array([ \
                rank_array[1,k] >= \
                    -np.log(1 - (1 - ((q*(k+1))/len(self.cells)))**(1/M[k]))
                for k in range(len(self.cells))])

        rank = np.linspace(1, len(self.cells), len(self.cells))
        # stores stuff back into self.cells
        for el in self.cells:
            el.flavin.rank = int(rank[rank_array[0] == el.cellid])
            el.flavin.classification = \
                    bool(classification[rank_array[0] == el.cellid])

    def summarise_classification(self):
        """ Generates resultsummary """
        df = pd.DataFrame({
            'category': [el.flavin.category for el in self.cells],
            'classification': [el.flavin.classification for el in self.cells]})
        self.resultsummary = pd.crosstab(df['category'], df['classification'])

    def add_cm(self):
        # rudimentary error handling
        if (self.resultsummary is None) or self.resultsummary.empty:
            self.summarise_classification()

        if self.resultsummary.empty:
            print('Please add classifications')
        else:
            # computes confusion matrix
            self.cm = np.array([ \
                [self.resultsummary.iloc[0,1], self.resultsummary.iloc[2,1]],
                [self.resultsummary.iloc[0,0], self.resultsummary.iloc[2,0]]])

    def add_fpr(self):
        if self.cm is None:
            self.add_cm()
        self.fpr = self.cm[0,1]/(self.cm[0,1]+self.cm[1,1])

    def add_tpr(self):
        if self.cm is None:
            self.add_cm()
        self.tpr = self.cm[0,0]/(self.cm[0,0]+self.cm[1,0])

    def add_fdr(self):
        if self.cm is None:
            self.add_cm()
        self.fdr = self.cm[0,1]/(self.cm[0,1]+self.cm[0,0])

    def add_accuracy(self):
        if self.cm is None:
            self.add_cm()
        self.accuracy = (self.cm[0,0]+self.cm[1,1])/ \
                (self.cm[0,0]+self.cm[0,1]+self.cm[1,0]+self.cm[1,1])
