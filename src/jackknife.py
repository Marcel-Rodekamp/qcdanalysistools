"""
    This file contains the
        * leave n out jackknife
        * leave n random out jackknife
    methods.
"""
import numpy as np
from numpy import average

def _leave_n_out(t_data, t_n = 1):
    """

        Returns: numpy.ndarray
            Leaves out t_n of the data set and evaluates t_avg on the sub data
    """
    subdata_sets = [None]*(t_data.shape[0]//t_n)

    for k in range(t_data.shape[0]//t_n):
        subdata_sets[k] = np.delete(
            t_data,
            [ k+i if k+i < t_data.shape[0] else i for i in range(t_n) ],
            axis = 0,
        )

    return np.array(subdata_sets)

def _leave_n_out_ran(t_data,t_num_ran_indices,t_n=1):

    subdata_sets = [None]*(t_num_ran_indices)

    leave_out_index_list = np.random.randint(0,high=t_data.shape[0]-1,size=t_num_ran_indices)

    for k in range(len(leave_out_index_list)):
        subdata_sets[k] = np.delete(
            t_data,
            [ leave_out_index_list[k]+i if leave_out_index_list[k]+i < t_data.shape[0] else i for i in range(t_n) ],
            axis = 0,
        )

    return np.array(subdata_sets)

def jackknife_var(t_data, t_n = 1, t_est = None, t_subdata_sets = None, t_random_leaveout = False, t_num_ran_indices=None, t_avg = np.average, **kwargs):
    if t_est == None:
        t_est = t_avg(t_data, axis = 0, **kwargs)

    if t_subdata_sets == None:
        if t_random_leaveout:
            t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices==None else t_num_ran_indices
            t_subdata_sets = _leave_n_out_ran(t_data,t_num_ran_indices=t_num_ran_indices, t_n = t_n)
        else:
            t_subdata_sets = _leave_n_out(t_data, t_n = t_n)

    var = 0
    for kth_est in t_subdata_sets:
        var += np.square(kth_est - t_est)

    return var * (t_data.shape[0]-1) / t_data.shape[0]

def jackknife_est(t_data, t_n = 1, t_est = None, t_subdata_sets = None, t_random_leaveout = False, t_num_ran_indices=None, t_avg = np.average, **kwargs):
    if t_est == None:
        t_est = t_avg(t_data, axis = 0, **kwargs)

    if t_subdata_sets == None:
        if t_random_leaveout:
            t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices==None else t_num_ran_indices
            t_subdata_sets = _leave_n_out_ran(t_data,t_num_ran_indices=t_num_ran_indices, t_n = t_n)
        else:
            t_subdata_sets = _leave_n_out(t_data, t_n = t_n)

    return t_data.shape[0]*t_est-(t_data.shape[0]-1)*t_avg(t_subdata_sets,axis=0, **kwargs)

def jackknife(t_data, t_n = 1, t_avg = np.average, t_random_leaveout = False, t_num_ran_indices=None, t_bootstraped = False, t_blocked = False, t_num_blocks = None, **kwargs):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable or raw data on
            which t_avg must be called. It is assumed, that axis=0 represents
            the data points of the method and subsequent axis' are assumed to
            represent multidimensional estimators.
        t_n: int, default: 1
            Size of leave n out method.
        t_avg: callable function, default: numpy.average
            Variation of an average function or observable which is called to
            determine the estimator on the (sub-)data sets.
        t_random_leaveout: bool, default: False
            Set to `True` if the leave out subdata sets should be determined with
            random drawn indices.
            If t_bootstraped is True, this will automatically be True.
        t_num_ran_indices: int, default: None
            Defines the number of random indices drawn in the random leave out
            method. If default is used it is determined to half data size.
        t_bootstraped: bool, default: False
            Set to `True` if the jackknife should be combined with statistical
            bootstrap i.e.
                * randomly drawn leave out indices
                * Jackknife bias improved estimator
                * Bootstrap Variance
        t_blocked: bool, default: False
            Set to `True` if the Jackknife should be combined with the blocking
            method i.e.
                * t_data is blocked into t_num_blocks subdata blocks
                * On each subblock a jackknife (combined with bootstrap) is performed
                * A simple average of jackknife estimators and variances is returned
            In that case `t_num_blocks` must not be `None`!
        t_num_blocks: int, list of ints
            A integer which defines the number of
        **kwargs:
            keyworded arguments which can be passed to t_avg. Note that
                axis
            from numpy is going to be overwritten. This is due to the assumption
            that axis=0 represents the data.

        Returns: numpy.ndarray, numpy.ndarray
            estimator, variance
    """
    # define return values
    est = None
    var = None

    if t_blocked and t_bootstraped:
        pass
    elif t_bootstraped:
        pass
    elif t_blocked:
        pass
    else:
        # simple leave n out (randomnized)
        est = jackknife_est(t_data, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices, t_avg = t_avg, **kwargs)
        var = jackknife_var(t_data, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices, t_avg = t_avg, **kwargs)

    return est,var
