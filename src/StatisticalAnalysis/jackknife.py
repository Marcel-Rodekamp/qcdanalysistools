r"""
    This file contains the
        * leave n out jackknife
        * leave n random out jackknife
    methods.
"""
import numpy as np

def _leave_n_out(t_data, t_n = 1):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
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
    # initialize tensor over subdata sets
    subdata_sets = np.zeros( shape=( t_data.shape[0]//t_n, t_data.shape[0]-t_n, *t_data.shape[1:] ) )

    for k in range(t_data.shape[0]//t_n):
        # delete t_n from the full data block and store it in the subdata matrix
        subdata_sets[k] = np.delete(
            t_data,
            [ k+i if k+i < t_data.shape[0] else i for i in range(t_n) ],
            axis = 0,
        )

    return subdata_sets

def _leave_n_out_ran(t_data,t_num_ran_indices,t_n=1):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
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

    # initialize tensor over subdata sets
    subdata_sets = np.zeros( shape=( t_num_ran_indices, t_data.shape[0]-t_n, *t_data.shape[1:] ) )

    # draw indices which should be left out
    leave_out_index_list = np.random.randint(0,high=t_data.shape[0]-1,size=t_num_ran_indices)

    for k in range(len(leave_out_index_list)):

        # delete t_n from the full data block started at a random index k and
        # store it in the subdata matrix
        subdata_sets[k] = np.delete(
            t_data,
            [ leave_out_index_list[k]+i if leave_out_index_list[k]+i < t_data.shape[0] else i for i in range(t_n) ],
            axis = 0,
        )

    return subdata_sets

def jackknife_est(t_data, t_n = 1, t_obs = np.average, t_random_leaveout = False, t_num_ran_indices=None, **obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data.
        t_n: int, default: 1
            Size of leave n out method.
        t_random_leaveout: bool, default: False
            Set to `True` if the leave out subdata sets should be determined with
            random drawn indices.
        t_num_ran_indices: int, default: None
            Defines the number of random indices drawn in the random leave out
            method. If default is used it is determined to half data size.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Estimator

        This functions uses the leave n out jackknife method in order to determine
        the estimator of the observable determined over t_data.
        Let $N$ be the size of the data set and $K = \frac{N}{t_n}$ be the number
        of subdata sets. Let $X = {x_i}_{i\in[N]}$ denote the total data set
            1. Compute Observable on Dataset          --> $\hat{\Theta}=\Theta(X)$
            2. On data X perform a leave n out method --> ${X_k}_{k\in[0,K-1]}$
            3. Compute observable on each subdata set --> $\Theta_k=\Theta(X_k)$
            4. Compute biased estimator               --> $\tilde{\Theta} = \frac{1}{K} \sum_{k\in[0,K-1]} \Theta_k$
            5. Compute biasimproved estimator         --> $\Theta_{est} = \hat{\Theta} - (K-1) (\tilde{\Theta} - \hat{\Theta})$
            6. Return $\Theta_{est}$
    """
    # 1. determine the estimator on the full data set
    Theta_hat = t_obs(t_data,**obs_kwargs)

    # 2. create leave n out data sets
    if t_random_leaveout:
        t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices is None else t_num_ran_indices
        subdata_sets = _leave_n_out_ran(t_data,t_num_ran_indices=t_num_ran_indices, t_n = t_n,)
    else:
        subdata_sets = _leave_n_out(t_data, t_n = t_n)

    K = subdata_sets.shape[0]

    # 3. Compute observables
    Theta_k = np.zeros( shape = (K,*t_data.shape[1:]) )
    for k,x_k in enumerate(subdata_sets):
        Theta_k[k] = t_obs(x_k,**obs_kwargs)

    # 4.
    Theta_tilde = np.average(Theta_k,axis=0)

    # determine and return the bias reduced estimator
    # the first (inner) average, averages in each block, the second (outer) does
    # over the subdata sets, for index details see _leave_n_out(_ran) documentation.
    return K * Theta_hat -(K-1) * Theta_tilde

def jackknife_var(t_data, t_obs = np.average, t_n = 1, t_random_leaveout = False, t_num_ran_indices = None,**obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data.
        t_n: int, default: 1
            Size of leave n out method.
        t_random_leaveout: bool, default: False
            Set to `True` if the leave out subdata sets should be determined with
            random drawn indices.
        t_num_ran_indices: int, default: None
            Defines the number of random indices drawn in the random leave out
            method. If default is used it is determined to half data size.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Estimator

        This functions uses the leave n out jackknife method in order to determine
        the estimator of the observable determined over t_data.
        Let $N$ be the size of the data set and $K = \frac{N}{t_n}$ be the number
        of subdata sets. Let $X = {x_i}_{i\in[N]}$ denote the total data set
            1. Compute Observable on Dataset          --> $\hat{\Theta}=\Theta(X)$
            2. On data X perform a leave n out method --> ${X_k}_{k\in[0,K-1]}$
            3. Compute observable on each subdata set --> $\Theta_k=\Theta(X_k)$
            4. Compute variance                       --> $\sigma^2 = \frac{K-1}{K} \sum_{k\in[0,K-1]} \left( \Theta_k - \hat{\Theta} \right)^2$
            6. Return $\sigma^2$
    """

    # 1. determine the estimator on the full data set
    Theta_hat = t_obs(t_data, **obs_kwargs)

    # 2. create leave n out data sets
    if t_random_leaveout:
        t_num_ran_indices = t_data.shape[0]//2 if t_num_ran_indices is None else t_num_ran_indices
        subdata_sets = _leave_n_out_ran(t_data,t_num_ran_indices=t_num_ran_indices, t_n = t_n)
    else:
        subdata_sets = _leave_n_out(t_data, t_n = t_n)

    K = subdata_sets.shape[0]

    # 3. Compute observables
    Theta_k = np.zeros( shape = (K,*t_data.shape[1:]) )
    for k,x_k in enumerate(subdata_sets):
        Theta_k[k] = t_obs(x_k,**obs_kwargs)

    # determine variance
    return ((K-1)/K) * np.sum( np.square(Theta_k-Theta_hat), axis = 0 )

def jackknife(t_data, t_obs = np.average, t_n = 1,  t_random_leaveout = False, t_num_ran_indices=None, t_blocked = False, t_num_blocks = None,**obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data of an observable. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data
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
            l_est[block_id] = jackknife_est(blocking.get_block(t_data,block_id,block_size), t_obs=t_obs, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices,**obs_kwargs)
            l_var[block_id] = jackknife_var(blocking.get_block(t_data,block_id,block_size), t_obs=t_obs, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices,**obs_kwargs)

        l_est[t_num_blocks-1] = jackknife_est(blocking.get_block(t_data,t_num_blocks-1,block_size,t_is_end=True),t_obs=t_obs, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices,**obs_kwargs)
        l_var[t_num_blocks-1] = jackknife_var(blocking.get_block(t_data,t_num_blocks-1,block_size,t_is_end=True),t_obs=t_obs, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices,**obs_kwargs)

        est = np.average(l_est,axis=0)
        var = np.average(l_var,axis=0)
    else:
        # simple leave n out (randomnized)
        est = jackknife_est(t_data, t_obs=t_obs, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices,**obs_kwargs)
        var = jackknife_var(t_data, t_obs=t_obs, t_n = t_n, t_random_leaveout = t_random_leaveout, t_num_ran_indices = t_num_ran_indices,**obs_kwargs)

    return est,var
