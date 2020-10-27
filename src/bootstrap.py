"""
"""
import numpy as np
from numpy import average

def _leave_out(t_data,t_num_ran_indices):
    # The bootstrap methods leaves n randomly chosen data points out.
    # This function can be recycled for this pupose. Therefore, the default case
    # t_n = 1 is used.
    from qcdanalysistools.jackknife import _leave_n_out_ran
    return _leave_n_out_ran(t_data,t_num_ran_indices,t_n=1)

def bootstrap_var(t_data, t_num_ran_indices = None, t_subdata_sets = None, t_est = None, t_avg = np.average, **kwargs):
    if t_est is None:
        t_est = bootstrap_est(t_data, t_subdata_sets, t_num_ran_indices, t_avg, **kwargs)

    if t_subdata_sets is None:
        t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices==None else t_num_ran_indices
        t_subdata_sets = _leave_out(t_data,t_num_ran_indices=t_num_ran_indices)

    var = 0
    for kth_est in t_subdata_sets:
        var += np.square(kth_est - t_est)

    return var / t_data.shape[0]

def bootstrap_est(t_data, t_subdata_sets = None, t_num_ran_indices=None, t_avg = np.average, **kwargs):
    if t_subdata_sets is None:
        t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices==None else t_num_ran_indices
        t_subdata_sets = _leave_out(t_data,t_num_ran_indices=t_num_ran_indices)

    return t_avg(t_subdata_sets, axis = 0, **kwargs)

def bootstrap(t_data, t_num_ran_indices = None, t_avg = np.average, t_blocked = False, t_num_blocks = None, **kwargs):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable or raw data on
            which t_avg must be called. It is assumed, that axis=0 represents
            the data points of the method and subsequent axis' are assumed to
            represent multidimensional estimators.
        t_avg: callable function, default: numpy.average
            Variation of an average function or observable which is called to
            determine the estimator on the (sub-)data sets.
        t_blocked: bool, default: False
            Set to `True` if the Bootstrap should be combined with the blocking
            method i.e.
                * t_data is blocked into t_num_blocks subdata blocks
                * On each subblock a bootstrap is performed
                * A simple average of the bootstrap estimators and variances is returned
            In that case `t_num_blocks` must not be `None`!
        t_num_blocks: int, list of ints
            An integer which defines the number of blocks in which t_data is blocked
            once the blocking method is applied.
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

    if t_blocked:
        pass
    else:
        # simple leave n out (randomnized)
        est = bootstrap_est(t_data, t_avg = t_avg, t_num_ran_indices = t_num_ran_indices, **kwargs)
        var = bootstrap_var(t_data, t_est = est, t_avg = t_avg, **kwargs)

    return est,var
