import numpy as np
import matplotlib.pylab as plt
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

version= 0.1

def estimateMI(data, overtime= True, n_bootstraps= 100, ci= [0.25, 0.75],
               Crange= None, gammarange= None, verbose= False):
    '''
    Estimates the mutual information between classes of time series.

    Uses sklean to optimise a pipeline for classifying the individual time series,
    choosing the number of PCA components (3-7), the classifier - a support vector
    machine with either a linear or a radial basis function kernel - and its C and
    gamma parameters.

    Errors are found using bootstrapped datasets.

    Parmeters:  data: list of arrays

                    A list of arrays, where each array comprises a collection of time series
                    from a different class and so the length of the list gives the number of
                    classes. For the array for one class, each row is a single-cell time
                    series.

                overtime: boolean (default: False)

                    If True, calculate the mutual information as a function of the duration of
                    the time series, by finding the mutuation information for all possible
                    sub-time series that start from t= 0.


                n_bootstraps: int, optional

                    The number of bootstraps used to estimate errors.


                ci: 1x2 array or list, optional

                    The lower and upper confidence intervals.

                    E.g. [0.25, 0.75] for the interquartile range


                Crange: array, optional

                    An array of potential values for the C parameter of the support vector machine
                    and from which the optimal value of C will be chosen.

                    If None, np.logspace(-3, 3, 10) is used. This range should be increased if
                    the optimal C is one of the boundary values.

                    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html


                gammarange: array, optional

                    An array of potential values for the gamma parameter for the radial basis
                    function kernel of the support vector machine and from which the optimal
                    value of gamma will be chosen.

                    If None, np.logspace(-3, 3, 10) is used. This range should be increased if
                    the optimal gamma is one of the boundary values.

                    See https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html


                verbose: boolean (default: False)

                    If True, display the results of internal steps, such as the optimal
                    parameters for the classifier.


    Returns:    res: array

                    Summary statistics from the bootstrapped datasets -- the median mutual
                    information and the 10% and 90% confidence limits.

                    If overtime is True, each row corresponds to a different duration of the time
                    series with the shortest duration, just the first time point, in the first row
                    and the longest duration, the entire time series, in the last row.


    '''
    # default values
    if not Crange: Crange= np.logspace(-3, 3, 10)
    if not gammarange: gammarange= np.logspace(-3, 3, 10)

    # data is a list with one array of time series for different class
    n_classes= len(data)
    Xo= np.vstack([timeseries for timeseries in data])
    y= np.hstack([i*np.ones(timeseries.shape[0]) for i, timeseries in enumerate(data)])

    if overtime:
        # loop over time series iteratively
        durations= np.arange(1, Xo.shape[1]+1)
    else:
        # full time series only
        durations= [Xo.shape[1]]
    # array for results
    res= np.zeros((len(durations), 3))

    for j, duration in enumerate(durations):
        # slice of of time series
        if verbose: print('duration of time series is', duration)
        X= Xo[:,:duration]

        # initialise scikit-learn routines
        nPCArange= range(1, X.shape[1]+1) if X.shape[1] < 7 else [3, 4, 5, 6, 7]
        params= [{'project__n_components': nPCArange},
                  {'classifier__kernel': ['linear'], 'classifier__C': Crange},
                  {'classifier__kernel': ['rbf'], 'classifier__C': Crange, 'classifier__gamma' : gammarange},
                  ]
        pipe= Pipeline([('project', PCA()),
                        ('rescale', StandardScaler()),
                        ('classifier', svm.SVC()),
                        ])

        # find best params for pipeline
        grid_pipeline= GridSearchCV(pipe, params, n_jobs= -1, cv= 5)
        grid_pipeline.fit(X, y)
        if verbose: print(grid_pipeline.best_estimator_)
        pipe.set_params(**grid_pipeline.best_params_)

        # find mutual information for each bootstrapped dataset
        mi= np.empty(n_bootstraps)
        for i in range(n_bootstraps):
            X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25)
            # run classifier use optimal params
            pipe.fit(X_train, y_train)
            y_predict= pipe.predict(X_test)
            # estimate mutual information
            p_y= 1/n_classes
            p_yhat_given_y= confusion_matrix(y_test, y_predict, normalize= 'true')
            p_yhat= np.sum(p_y*p_yhat_given_y, 0)
            h_yhat= -np.sum(p_yhat[p_yhat > 0]*np.log2(p_yhat[p_yhat > 0]))
            log2_p_yhat_given_y= np.ma.log2(p_yhat_given_y).filled(0)
            h_yhat_given_y= -np.sum(p_y*np.sum(p_yhat_given_y*log2_p_yhat_given_y, 1))
            mi[i]= h_yhat - h_yhat_given_y

        # summary statistics - median and confidence intervals
        res[j,:]= [np.median(mi), np.sort(mi)[int(np.min(ci)*n_bootstraps)],
                   np.sort(mi)[int(np.max(ci)*n_bootstraps)]]
        if verbose: print(f'median MI= {res[j,0]:.2f} [{res[j,1]:.2f}, {res[j,2]:.2f}]')
    return res



def plotMIovertime(res, color= 'b', label= None):
    '''
    Plots the median mutual information against the duration of the time series.
    The region between the lower and upper confidence limits is shaded.

    Parameters:

        res: array

            An array of the median mutual information and the lower and upper confidence
            limits with the first row corresponding to the time series of shortest
            duration and the last two corresponding to the time series of longest
            duration.


        color: string, optional

            A Matplotlib color for both the median and shaded interquartile range.


        label: string, optional

            A label for the legend.

    '''
    durations= np.arange(res.shape[0])
    plt.plot(durations, res[:,0], '.-', color= color, label= label)
    plt.fill_between(durations, res[:,1], res[:,2], color= color, alpha= 0.2)
