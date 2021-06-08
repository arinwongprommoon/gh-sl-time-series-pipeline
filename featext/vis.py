#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.stats as st
import catch22

# when I inevitably re-structure/re-factor stuff in a bit, this will go to a
# file that makes more sense
catch22features = [
'DN_HistogramMode_5',
'DN_HistogramMode_10',
'CO_f1ecac',
'CO_FirstMin_ac',
'CO_HistogramAMI_even_2_5',
'CO_trev_1_num',
'MD_hrv_classic_pnn40',
'SB_BinaryStats_mean_longstretch1',
'SB_TransitionMatrix_3ac_sumdiagcov',
'PD_PeriodicityWang_th0_01',
'CO_Embed2_Dist_tau_d_expfit_meandiff',
'IN_AutoMutualInfoStats_40_gaussian_fmmi',
'FC_LocalSimple_mean1_tauresrat',
'DN_OutlierInclude_p_001_mdrmd',
'DN_OutlierInclude_n_001_mdrmd',
'SP_Summaries_welch_rect_area_5_1',
'SB_BinaryStats_diff_longstretch0',
'SB_MotifThree_quantile_hh',
'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
'SP_Summaries_welch_rect_centroid',
'FC_LocalSimple_mean3_stderr'
]

class KS2:
    """
    Container, re-structures output class of scipy.stats.ks_2samp, which tests
    whether two samples are drawn from the same distribution

    Parameters:
    -----------
    feature_name = string,
        name of catch22 feature
    dist1 = list of floats,
        catch22 values for first sample
    dist2 = list of floats,
        catch22 values for second sample

    Attributes:
    -----------
    feature_name
    statistic = float,
        test statistic for two-sample Kolmogorov-Smirnoff test
    pvalue = float,
        p-value for two-sample Kolmogorov-Smirnoff test
    """
    def __init__(self, feature_name, dist1, dist2):
        self.feature_name = feature_name

        KstestResult = st.ks_2samp(dist1, dist2)
        self.statistic = KstestResult.statistic
        self.pvalue = KstestResult.pvalue

class TS_TopFeatures_ks2:
    """
    Results of two-sample Kolmogorov-Smirnoff test on each catch22 feature
    from two lists of CellAttr objects
    Method is a parallel to hctsa TS_TopFeatures(), but not adapted from it

    Parameters:
    -----------
    list_CellAttr_1 = list of pipeline.CellAttr objects
        first list of cells
    list_CellAttr_2 = list of pipeline.CellAttr objects
        second list of cells

    Attributes:
    -----------
    list_KS2 = list of KS2 objects,
        stores feature names and results of Kolmogorov-Smirnoff test
    """
    def __init__(self, list_CellAttr_1, list_CellAttr_2):
        list_KS2 = []
        for feature_idx, feature_name in enumerate(catch22features):
            list_KS2.append(KS2(feature_name = feature_name,
                dist1 = [cell.hctsa_vec[feature_idx] for cell in list_CellAttr_1],
                dist2 = [cell.hctsa_vec[feature_idx] for cell in list_CellAttr_2]))
        self.list_KS2 = list_KS2

    def get_top_features(self, threshold = 0.05):
        """
        Outputs top features, with p-values below a threshold (default 0.05)
        to a pandas dataframe
        """
        top_features_df = \
                pd.DataFrame({'feature_name': [feature.feature_name
                                  for feature in self.list_KS2],
                 'pvalue': [feature.pvalue
                            for feature in self.list_KS2]})
        top_features_df = top_features_df.sort_values(by = ['pvalue'])
        top_features_df = top_features_df[top_features_df.pvalue < threshold]
        return top_features_df
