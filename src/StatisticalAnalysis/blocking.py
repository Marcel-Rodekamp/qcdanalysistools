import numpy as np

def get_block(t_data,t_block_id,t_block_size,t_is_end=False):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_block_id: int
            The number of the block which should be returned.
        t_block_size: int
            The size of the blocks in which t_data becomes blocked.
        t_is_end: bool, default: False
            If this is set to True the remainder of the data is returned, starting
            at the index t_block_id*t_block_size.

        Returns: numpy.ndarray
            Data block. Let d be the total dimension of the estimator,
            n_i denotes the number of elements in that particular dimension and
            K = t_num_blocks be the number of blocks, then
                subdata_sets.shape = (t_block_size,n_1,n_2,...,n_d)
                                   = (t_block_size,t_data.shape[1:])

        This functions returns a slice of the data t_data along the 0th axis.
    """
    if t_is_end:
        return t_data[t_block_id*t_block_size : ]
    else:
        return t_data[t_block_id*t_block_size : (t_block_id+1)*t_block_size]

def blocking_data(t_data,t_num_blocks):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_num_blocks: int
            Number of blocks in which t_data becomes devided

        Returns: numpy.ndarray
            Set of data blocks. Let d be the total dimension of the estimator,
            n_i denotes the number of elements in that particular dimension and
            K = t_num_blocks be the number of blocks, then
                subdata_sets.shape = (K,n_1,n_2,...,n_d) = (K,t_data.shape[1:])

        This functions blocks the data in K = t_num_blocks blocks.
    """

    subdata_sets = [None]*t_num_blocks

    block_size = t_data.shape[0] // t_num_blocks

    # slice the data in subblocks
    for block_id in range(t_num_blocks):
        subdata_sets[block_id] = get_block(t_data,block_id,block_size)

    return np.array(subdata_sets)

def blocking_est(t_data, t_obs = np.average, t_num_blocks = 2,**obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data which becomes blocked and processed. It is assumed that axis = 0
            represents the different data points in the set and all other axis'
            account for the dimensionality of the estimator.
        t_num_blocks: int
            Number of blocks in which t_data becomes devided
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data.
        Returns: numpy.ndarray
            Estimator
        This function computes the estimator on the given t_data by blocking the
        data first and then averaing over all blocks.
        Let $N$ be the data size and $K$ be the number of blocks
            1. Block the dataset                          --> ${X_k}_{k\in[0,K-1]}$
            2. Compute the observable on each subdata set --> $\Theta_k = \Theta(X_k)$
            3. Compute the Estimator                      --> $\tilde{\Theta} = \frac{1}{K} \sum_{k\in[0,K-1]} \Theta_k$
            4. Return the Testimator

    """
    # 1. get blocked data
    blocked_data = blocking_data(t_data,t_num_blocks=t_num_blocks)

    # 2. Compute observables
    Theta_k = np.zeros( shape = (t_num_blocks,*t_data.shape[1:]) )
    for k,x_k in enumerate(blocked_data):
        Theta_k[k] = t_obs(x_k,**obs_kwargs)

    # 3. Compute estimator
    return np.average( Theta_k, axis = 0 )

def blocking_var(t_data, t_obs = np.average, t_num_blocks = 2,**obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data which becomes blocked and processed. It is assumed that axis = 0
            represents the different data points in the set and all other axis'
            account for the dimensionality of the estimator.
        t_num_blocks: int
            Number of blocks in which t_data becomes devided
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data.
        Returns: numpy.ndarray
            Variance
        This function computes the estimator on the given t_data by blocking the
        data first and then averaing over all blocks.
        Let $N$ be the data size and $K$ be the number of blocks
            1. Block the dataset                          --> ${X_k}_{k\in[0,K-1]}$
            2. Compute the observable on each subdata set --> $\Theta_k = \Theta(X_k)$
            3. Compute the estimator                      --> $\tilde{\Theta} = \frac{1}{K} \sum_{k\in[0,K-1]} \Theta_k$
            4. Compute the variance                       --> $\sigma^2 = \frac{1}{K} \sum_{k\in[0,K-1]} \left(\Theta_k - \tilde{\Theta}\right)^2$
            5. Return the variance

    """
    # 1. get blocked data
    blocked_data = blocking_data(t_data,t_num_blocks=t_num_blocks)

    # 2. Compute observables
    Theta_k = np.zeros( shape = (t_num_blocks,*t_data.shape[1:]) )
    for k,x_k in enumerate(blocked_data):
        Theta_k[k] = t_obs(x_k,**obs_kwargs)

    # 3,4. Compute estimator
    return np.var( Theta_k, axis = 0 )

def blocking(t_data, t_obs = np.average, t_num_blocks = 2,**obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data which becomes blocked and processed. It is assumed that axis = 0
            represents the different data points in the set and all other axis'
            account for the dimensionality of the estimator.
        t_num_blocks: int
            Number of blocks in which t_data becomes devided
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data.
        Returns: numpy.ndarray
            Estimator, Variance
    """
    return blocking_est(t_data,t_obs=t_obs,t_num_blocks=t_num_blocks,**obs_kwargs),\
           blocking_var(t_data,t_obs=t_obs,t_num_blocks=t_num_blocks,**obs_kwargs)
