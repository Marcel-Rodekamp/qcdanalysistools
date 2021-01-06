r"""
    This file contains the
        * Bootstrap
    method.

        If a more involved estimator of data needs to be computed (such as a fit),
        e.g. the data needs to be processed on each subdata set etc, two ways are
        implemented:
            * If the output of a observable is just the estimator one can pass
              the function computing it via the `t_obs` and `**obs_kwargs` arguments
              to one of the ready to go methods.
            * If more then the estimator is important for further analysis (p-value,
              chi², etc.):
              This can not be implemented without making assumptions on the output
              of the observable function. Therefore, methods have been implemented to
              serve the following skeleton:

        ```
        # 0. import & read in
        import numpy as np
        import qcdanalysistools.analysis.BlockingParams
        from qcdanalysistools.analysis.Blocking import *

        full_data = ...

        # 1. Create parameters to specify jackknife
        bst_params = BlockingParams(t_data_size,t_num_subdatasets,t_with_blocking,t_num_blocks)

        # 2. Define observable/estimator function acting on one subdata set
        def obs(t_kth_subdataset,*args,**kwargs):
            ...

        # 3. Compute the estimators for each subdata set
        obs_out = [None]*bst_params.num_subdatasets
        Theta_k = np.zeros(shape = (bst_params.num_subdatasets,...))

        for k in range(bst_params.num_subdatasets):
            obs_out[k] = obs(subdataset(full_data,k,bst_params),*args,**kwargs)
            Theta_k[k] = obs_out[k].estimated_quantity # or however one accesses it

        # 4. Compute the biased improved estimator and variance
        est = skeleton_est(Theta_k,bst_params)
        var = skeleton_var(Theta_k,bst_params)

        # 5. Output
        ...

        ```
"""
import numpy as np
from ..analysisParams import BlockingParams

# ==============================================================================
# Assertion based on the parameter
# ==============================================================================
def BlockingAssert(t_params):
    if not isinstance(t_params,BlockingParams):
        raise RuntimeError(f"Blocking does not work with parameter of type {t_params.analysis_type}.")

# ==============================================================================
# Creating Data Subsets Following the Blocking method
# ==============================================================================

def get_block(t_data,t_block_id,t_params):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_block_id: int
            The number of the block which should be returned.
        t_params: BlockingParams

        Returns: numpy.ndarray
            Data block. Let d be the total dimension of the estimator,
            n_i denotes the number of elements in that particular dimension and
            K = t_num_blocks be the number of blocks, then
                subdata_sets.shape = (t_block_size,n_1,n_2,...,n_d)
                                   = (t_block_size,t_data.shape[1:])

        This functions returns a slice of the data t_data along the 0th axis.
    """
    BlockingAssert(t_params)

    if t_block_id == t_params.num_blocks - 1:
        return t_data[t_block_id*t_params.block_size : ]
    else:
        return t_data[t_block_id*t_params.block_size : (t_block_id+1)*t_params.block_size]

# To provide interchangability define the alias
subdataset = get_block

def blocking_data(t_data,t_params):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: BlockingParams

        Returns: numpy.ndarray
            Data block. Let d be the total dimension of the estimator,
            n_i denotes the number of elements in that particular dimension and
            K = t_num_blocks be the number of blocks, then
                subdata_sets.shape = (t_block_size,n_1,n_2,...,n_d)
                                   = (t_block_size,t_data.shape[1:])

        This functions returns a slice of the data t_data along the 0th axis.
    """
    BlockingAssert(t_params)

    subdata_sets = [None]*t_params.num_blocks

    # slice the data in subblocks
    for block_id in range(t_params.num_blocks):
        subdata_sets[block_id] = get_block(t_data,block_id,t_params)

    return np.array(subdata_sets)

# To provide interchangability define the alias
subdatasets = blocking_data

# ==============================================================================
# Bits and Pieces
# ==============================================================================

# only wrappers
def est_k(t_data, t_params, t_k, t_obs = np.average, **obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Array, containing the full data set. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: JackknifeParams
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data. This can be useful if
            the t_obs outputs only the required estimators in each subdata set.
        t_k: int, default: None
            Determines which data sub set is taken.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Estimator

        This functions computes the estimator on the kth data supset
    """
    BlockingAssert(t_params)

    return t_obs(get_block(t_data, t_k, t_params),**obs_kwargs)

def skeleton_est(t_estimators,t_params):
    r"""
        t_estimators: numpy.ndarray
            Estimators computed on each subdata set in the jackknife method.
        t_params: JackknifeParams
    """
    BlockingAssert(t_params)

    if not isinstance(t_estimators, np.ndarray):
        try:
            t_estimators = np.array(t_estimators)
        except:
            raise ValueError(f"The estimators need to be of list type but are ({type(t_estimators)}).")

    if t_estimators.shape[0] != t_params.num_blocks:
        raise ValueError(f"The estimators on subdata sets are not valid, require axis 0 of length ({t_params.num_blocks}) but got (t_estimators.shape[0]).")

    return np.average(t_estimators,axis=0)

def skeleton_var(t_estimators,t_params):
    r"""
        t_estimators: numpy.ndarray
            Estimators computed on each subdata set in the jackknife method.
        t_params: JackknifeParams
    """
    BlockingAssert(t_params)

    if not isinstance(t_estimators, np.ndarray):
        try:
            t_estimators = np.array(t_estimators)
        except:
            raise ValueError(f"The estimators need to be of list type but are ({type(t_estimators)}).")

    if t_estimators.shape[0] != t_params.num_blocks:
        raise ValueError(f"The estimators on subdata sets are not valid, require axis 0 of length ({t_params.num_blocks}) but got (t_estimators.shape[0]).")

    return np.var(t_estimators,axis=0)

# ==============================================================================
# Ready to go estimator
# ==============================================================================

def est(t_data, t_params, t_obs = np.average, **obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: JackknifeParams
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data. This can be useful if
            the t_obs outputs only the required estimators in each subdata set.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

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
    BlockingAssert(t_params)

    # 1. & 2.
    Theta_k = [None]*t_params.num_blocks
    for k in range(t_params.num_blocks):
        Theta_k[k] = est_k(t_data,t_params,k,t_obs,**obs_kwargs)

    # 3. & 4.
    return skeleton_est(Theta_k,t_params)

def var(t_data, t_params, t_obs = np.average, **obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: JackknifeParams
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data. This can be useful if
            the t_obs outputs only the required estimators in each subdata set.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

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
    BlockingAssert(t_params)

    # 2. & 3.
    Theta_k = [None]*t_params.num_blocks
    for k in range(t_params.num_blocks):
        Theta_k[k] = est_k(t_data, t_params, k, t_obs, **obs_kwargs)

    # 4. & 5.
    return skeleton_var(Theta_k,t_params)

# ==============================================================================
# Ready to go Jackknife method computing estimator and variance
# ==============================================================================

def blocking(t_data, t_params, t_obs = np.average, **obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: JackknifeParams
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data. This can be useful if
            the t_obs outputs only the required estimators in each subdata set.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray, numpy.ndarray
            estimator, variance
    """
    BlockingAssert(t_params)

    Theta_k = [None]*t_params.num_blocks
    for k in range(t_params.num_blocks):
        Theta_k[k] = est_k(t_data, t_params, k, t_obs, **obs_kwargs)

    return skeleton_est(Theta_k,t_params),skeleton_var(Theta_k,t_params)
