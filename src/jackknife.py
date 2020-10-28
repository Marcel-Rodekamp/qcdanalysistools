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
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_n: int, default: 1
            Size of leave n out method.

        Returns: numpy.ndarray
            Returns a set of subdata sets where each subdata set contains N-t_n
            points that is the total size of t_data minus the amount of left out
            t_n data points.
            Let i index the number of subdata sets for which t_n points are left
            out and j index the remaining points within a subdata set. Further,
            N be the total data size, n = t_n.
            Then

                [subdata_{j=0}]_{i=0} [subdata_{j=0}]_{i=1} ... [subdata_{j=0}]_{i=N//n}
                [subdata_{j=1}]_{i=0} [subdata_{j=1}]_{i=1} ... [subdata_{j=1}]_{i=N//n}
                           .                      .                        .
                           .                      .                        .
                           .                      .                        .
                [subdata_{j=N-n}]_{i=0} [subdata_{j=N-n}]_{i=1} ... [subdata_{j=N-n}]_{i=N//n}

            => subdata_sets[j][i]

            Note that each element is of the dimension of the estimator i.e.
                t_data.shape[1:]

    """
    subdata_sets = np.zeros( shape=( t_data.shape[0]//t_n, t_data.shape[0]-t_n, *t_data.shape[1:] ) )

    for k in range(t_data.shape[0]//t_n):
        subdata_sets[k] = np.delete(
            t_data,
            [ k+i if k+i < t_data.shape[0] else i for i in range(t_n) ],
            axis = 0,
        )


    return subdata_sets

def _leave_n_out_ran(t_data,t_num_ran_indices,t_n=1):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_num_ran_indices: int, default: None
            Defines the number of random indices drawn in the random leave out
            method. If default is used it is determined to half data size.
        t_n: int, default: 1
            Size of leave n out method.

        Returns: numpy.ndarray
            Returns a set of subdata sets where each subdata set contains N-t_n
            points that is the total size of t_data minus the amount of left out
            t_n data points.
            Let i index the number of subdata sets for which t_n points are left
            out and j index the remaining points within a subdata set. Further,
            N be the total data size, n = t_n, and M = t_num_ran_indices
            Then

                [subdata_{j=0}]_{i=0} [subdata_{j=0}]_{i=1} ... [subdata_{j=0}]_{i=M}
                [subdata_{j=1}]_{i=0} [subdata_{j=1}]_{i=1} ... [subdata_{j=1}]_{i=M}
                           .                      .                        .
                           .                      .                        .
                           .                      .                        .
             [subdata_{j=N-n}]_{i=0} [subdata_{j=N-n}]_{i=1} ... [subdata_{j=N-n}]_{i=M}

            => subdata_sets[j][i]

            Note that each element is of the dimension of the estimator i.e.
                t_data.shape[1:]
    """

    subdata_sets = np.zeros( shape=( t_num_ran_indices, t_data.shape[0]-t_n, *t_data.shape[1:] ) )

    leave_out_index_list = np.random.randint(0,high=t_data.shape[0]-1,size=t_num_ran_indices)

    for k in range(len(leave_out_index_list)):
        subdata_sets[k] = np.delete(
            t_data,
            [ leave_out_index_list[k]+i if leave_out_index_list[k]+i < t_data.shape[0] else i for i in range(t_n) ],
            axis = 0,
        )

    return subdata_sets

def jackknife_est(t_data, t_n = 1, t_random_leaveout = False, t_num_ran_indices=None):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_n: int, default: 1
            Size of leave n out method.
        t_random_leaveout: bool, default: False
            Set to `True` if the leave out subdata sets should be determined with
            random drawn indices.
        t_num_ran_indices: int, default: None
            Defines the number of random indices drawn in the random leave out
            method. If default is used it is determined to half data size.

        Returns: numpy.ndarray
            Estimator

        This functions uses the leave n out jackknife method in order to determine
        the estimator of the data set t_data. Let N be the size of the data set
        and K be the number of subdata sets. Let
            Theta = 1/N sum_{n=1}^N x
        be the estimator on the total data set with data points x and Theta_k the
        one on the kth subdata set. Then the returned estimator is biased
        improved by
            N * Theta - (N-1) * 1/K sum_{k=1}^K Theta_k
    """

    # determine the estimator on the full data set
    global_est = np.average(t_data, axis = 0)

    # create leave n out data sets
    if t_random_leaveout:
        t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices is None else t_num_ran_indices
        subdata_sets = _leave_n_out_ran(t_data,t_num_ran_indices=t_num_ran_indices, t_n = t_n)
    else:
        subdata_sets = _leave_n_out(t_data, t_n = t_n)

    # determine and return the bias reduced estimator
    # the first (inner) average, averages in each block, the second (outer) does
    # over the subdata sets, for index details see _leave_n_out(_ran) documentation.
    return t_data.shape[0] * global_est \
         -(t_data.shape[0]-1) * np.average(np.average(subdata_sets,axis = 1),axis=0)

def jackknife_var(t_data, t_n = 1, t_random_leaveout = False, t_num_ran_indices = None):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_n: int, default: 1
            Size of leave n out method.
        t_random_leaveout: bool, default: False
            Set to `True` if the leave out subdata sets should be determined with
            random drawn indices.
        t_num_ran_indices: int, default: None
            Defines the number of random indices drawn in the random leave out
            method. If default is used it is determined to half data size.

        Returns: numpy.ndarray
            variance

        This functions uses the leave n out jackknife method in order to determine
        the variance of the data set t_data. Let N be the size of the data set and
        K be the number of subdata sets
            var = (N-1)/N * sum_{k=1}^K (Theta_k - Theta)^2
        where Theta_k is the estimator on the kth subdata set and
            Theta = 1/N sum_{n=1}^N x
        is the estimator on the total data set with data points x.
    """

    # determine the estimator on the full data set
    global_est = np.average(t_data, axis = 0)

    # create leave n out data sets
    if t_random_leaveout:
        t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices is None else t_num_ran_indices
        subdata_sets = _leave_n_out_ran(t_data,t_num_ran_indices=t_num_ran_indices, t_n = t_n)
    else:
        subdata_sets = _leave_n_out(t_data, t_n = t_n)

    # determine estimator on each subdata set
    est = np.average(subdata_sets, axis=0)

    # determine variance
    return ((t_data.shape[0]-1)/t_data.shape[0]) * np.sum( np.square(est-global_est), axis = 0 )

def jackknife(t_data, t_n = 1, t_random_leaveout = False, t_num_ran_indices=None, t_blocked = False, t_num_blocks = None):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_n: int, default: 1
            Size of leave n out method.
        t_random_leaveout: bool, default: False
            Set to `True` if the leave out subdata sets should be determined with
            random drawn indices.
        t_num_ran_indices: int, default: None
            Defines the number of random indices drawn in the random leave out
            method. If default is used it is determined to half data size.
        t_blocked: bool, default: False
            Set to `True` if the Jackknife should be combined with the blocking
            method i.e.
                * t_data is blocked into t_num_blocks subdata blocks
                * On each subblock a jackknife is performed
                * A simple average of jackknife estimators and variances over all
                  blocks is returned
            In that case `t_num_blocks` must not be `None`!
        t_num_blocks: int, list of ints
            A integer which defines the number of blocks in which the data is
            partitioned, see blocking method.

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
            l_est[block_id] = jackknife_est(blocking.get_block(t_data,block_id,block_size), t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices)
            l_var[block_id] = jackknife_var(blocking.get_block(t_data,block_id,block_size), t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices)

        l_est[t_num_blocks-1] = jackknife_est(blocking.get_block(t_data,t_num_blocks-1,block_size,t_is_end=True), t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices)
        l_var[t_num_blocks-1] = jackknife_var(blocking.get_block(t_data,t_num_blocks-1,block_size,t_is_end=True), t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices)

        est = np.average(l_est,axis=0)
        var = np.average(l_var,axis=0)
    else:
        # simple leave n out (randomnized)
        est = jackknife_est(t_data, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices)
        var = jackknife_var(t_data, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices)

    return est,var
