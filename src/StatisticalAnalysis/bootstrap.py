"""
"""
import numpy as np
from numpy import average

def _leave_out(t_data,t_num_ran_indices):
    """
        t_data: numpy.ndarray
            Data which becomes bootstraped. It is assumed that axis = 0
            represents the different data points in the set and all other axis'
            account for the dimensionality of the estimator.
        t_num_leave_outs: int
            Defines how many subdata sets are created by bootstrap.

        Returns:
            A set of subdata where each subdata is the original data set but without
            one randomly chosen data point
    """
    # The bootstrap methods leaves 1 randomly chosen data point out.
    # Following function can be recycled for this pupose. Therefore, the
    # default case t_n = 1 is used.
    from qcdanalysistools.jackknife import _leave_n_out_ran
    return _leave_n_out_ran(t_data,t_num_ran_indices,t_n=1)

def bootstrap_est(t_data,t_num_leave_outs):
    """
        t_data: numpy.ndarray
            Data which becomes bootstraped. It is assumed that axis = 0
            represents the different data points in the set and all other axis'
            account for the dimensionality of the estimator.
        t_num_leave_outs: int
            Defines how many subdata sets are created by bootstrap.

        Returns: numpy.ndarray
            Variance after bootstrapping for each dimension of the estimator. Let
            d be the total dimension of the estimator, n_i denotes the number
            of elements in that particular dimension.
                var.shape = (n_1,n_2,...,n_d) = t_data.shape[1:]

        This functions leaves a data point, which is chosen randomly, out (bootstrap)
        Then, the estimator (numpy.average) on each of these subdata sets is evaluated.
        The variance is determined by
            var = 1/N sum_{k=1}^K (Theta_k - Theta)^2
        where N is the data size, Theta_k is the estimator on the kth subdata set
        and
            Theta = 1/K sum_{k=1}^K Theta_k
    """
    # create subdata sets
    t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices==None else t_num_ran_indices
    subdata_sets = _leave_out(t_data,t_num_ran_indices=t_num_leave_outs)

    # determine and return the estimator
    # the first (inner) average, averages in each block, the second (outer) does
    # over the subdata sets, for index details see jackknife._leave_n_out_ran
    # documentation.
    return np.average(np.average(subdata_sets, axis = 1), axis = 0)

def bootstrap_var(t_data,t_num_leave_outs):
    """
        t_data: numpy.ndarray
            Data which becomes bootstraped. It is assumed that axis = 0
            represents the different data points in the set and all other axis'
            account for the dimensionality of the estimator.
        t_num_leave_outs: int
            Defines how many subdata sets are created by bootstrap.

        Returns: numpy.ndarray
            Variance after bootstrapping for each dimension of the estimator. Let
            d be the total dimension of the estimator, n_i denotes the number
            of elements in that particular dimension.
                var.shape = (n_1,n_2,...,n_d) = t_data.shape[1:]

        This functions leaves a data point, which is chosen randomly, out (bootstrap)
        Then, the estimator (numpy.average) on each of these subdata sets is evaluated.
        The variance is determined by
            var = 1/N sum_{k=1}^K (Theta_k - Theta)^2
        where N is the data size, Theta_k is the estimator on the kth subdata set
        and
            Theta = 1/K sum_{k=1}^K Theta_k
    """
    # create subdata sets
    t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices==None else t_num_ran_indices
    subdata_sets = _leave_out(t_data,t_num_ran_indices=t_num_leave_outs)

    # average in each block for index details see jackknife._leave_n_out_ran
    # documentation.
    return np.var( np.average( subdata_sets, axis = 1 ), axis = 0 )

def bootstrap(t_data,t_num_leave_outs, t_blocked = False, t_num_blocks = None):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable or raw data on
            which t_avg must be called. It is assumed, that axis=0 represents
            the data points of the method and subsequent axis' are assumed to
            represent multidimensional estimators.
        t_num_leave_outs: int
            Defines how many subdata sets are created by bootstrap.
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

        Returns: numpy.ndarray, numpy.ndarray
            estimator, variance
    """
    # define return values
    est = None
    var = None

    if t_blocked:
        import qcdanalysistools.blocking as blocking
        block_size = t_data.shape[0] // t_num_blocks
        l_est = [None]*t_num_blocks
        l_var = [None]*t_num_blocks

        for block_id in range(0,t_num_blocks):
            l_est[block_id] = bootstrap_est(blocking.get_block(t_data,block_id,block_size),t_num_leave_outs=t_num_leave_outs)
            l_var[block_id] = bootstrap_var(blocking.get_block(t_data,block_id,block_size),t_num_leave_outs=t_num_leave_outs)

        l_est[t_num_blocks-1] = bootstrap_est(blocking.get_block(t_data,t_num_blocks-1,block_size),t_num_leave_outs=t_num_leave_outs)
        l_var[t_num_blocks-1] = bootstrap_var(blocking.get_block(t_data,t_num_blocks-1,block_size),t_num_leave_outs=t_num_leave_outs)

        est = np.average(l_est,axis=0)
        var = np.average(l_var,axis=0)
    else:
        # simple leave n out (randomnized)
        est = bootstrap_est(t_data,t_num_leave_outs=t_num_leave_outs)
        var = bootstrap_var(t_data,t_num_leave_outs=t_num_leave_outs)

    return est,var