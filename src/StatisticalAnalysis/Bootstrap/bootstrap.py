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
        import qcdanalysistools.analysis.BootstrapParams
        from qcdanalysistools.analysis.Bootstrap import *

        full_data = ...

        # 1. Create parameters to specify jackknife
        bst_params = BootstrapParams(t_data_size,t_num_subdatasets,t_with_blocking,t_num_blocks)

        # 2. Define observable/estimator function acting on one subdata set
        def obs(t_kth_subdataset,*args,**kwargs):
            ...

        # 3. Compute the estimators for each subdata set
        # Note if blocking is required this mus be implemented here aswell!
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
from ..analysisParams import BootstrapParams

# ==============================================================================
# Assertion based on the parameter
# ==============================================================================
def BootstrapAssert(t_params):
    if not isinstance(t_params,BootstrapParams):
        raise RuntimeError(f"Bootstrap does not work with parameter of type {t_params.analysis_type}.")

# ==============================================================================
# Creating Data Subsets following the Jackknife, leave n out, methods
# ==============================================================================

def subdataset(t_data, t_params):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: JackknifeParams
            Parameter of the jackknife method

        Returns: numpy.ndarray
            Returns the kth subdata set containing N points that is the
            total size of t_data.
    """
    BootstrapAssert(t_params)
    return np.take(t_data,[np.random.randint(0,high=t_params.data_size) for _ in range(t_params.data_size)],axis=0)

def subdatasets(t_data,t_params):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: JackknifeParams
            Parameter of the jackknife method

        Returns: numpy.ndarray
            Returns the subdatasets each containing N points that is the
            total size of t_data.
    """
    subdatasets = np.zeros( shape=( t_params.num_subdatasets, t_params.data_size, *t_data.shape[1:] ) )

    for k in range(t_params.num_subdatasets):
        subdatasets[k] = subdataset(t_data,k,t_params)

# ==============================================================================
# Bits and Pieces
# ==============================================================================

def est_k(t_data, t_params, t_obs = np.average, **obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Array, containing the full data set. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: BootstrapParams
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data. This can be useful if
            the t_obs outputs only the required estimators in each subdata set.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Estimator

        This functions computes the estimator on the kth data supset
    """
    BootstrapAssert(t_params)

    return t_obs(subdataset(t_data,t_params),**obs_kwargs)

def skeleton_est(t_estimators,t_params):
    r"""
        t_estimators: numpy.ndarray
            Estimators computed on each subdata set in the jackknife method.
        t_biased_est: numpy.ndarray or float
            Biased estimator coming from a estimation on the full data set.
        t_params: BootstrapParams
    """
    BootstrapAssert(t_params)

    if not isinstance(t_estimators, np.ndarray):
        try:
            t_estimators = np.array(t_estimators)
        except:
            raise ValueError(f"The estimators need to be of list type but are ({type(t_estimators)}).")

    if t_estimators.shape[0] != t_params.num_subdatasets:
        raise ValueError(f"The estimators on subdata sets are not valid, require axis 0 of length ({t_params.num_subdatasets}) but got (t_estimators.shape[0]).")

    return np.average(t_estimators,axis=0)

def skeleton_var(t_estimators,t_params):
    r"""
        t_estimators: numpy.ndarray
            Estimators computed on each subdata set in the jackknife method.
        t_params: BootstrapParams
    """
    BootstrapAssert(t_params)

    if not isinstance(t_estimators, np.ndarray):
        try:
            t_estimators = np.array(t_estimators)
        except:
            raise ValueError(f"The estimators need to be of list type but are ({type(t_estimators)}).")

    if t_estimators.shape[0] != t_params.num_subdatasets:
        raise ValueError(f"The estimators on subdata sets are not valid, require axis 0 of length ({t_params.num_subdatasets}) but got ({t_estimators.shape[0]}).")

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
        t_params: BootstrapParams
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data. This can be useful if
            the t_obs outputs only the required estimators in each subdata set.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Estimator

        This function uses the bootstrap method to determine the estimator of the
        observable determined over t_data.
        Let $N$ be the size of the data set. Let $X = {x_i}_{i\in[N]}$ denote
        the total data set and K be the number of subdata_sets
            1. On data X perform BootstrapParams      --> ${X_k}_{k\in[0,K-1]}$
            2. Compute observable on each subdata set --> $\Theta_k=\Theta(X_k)$
            3. Compute estimator                      --> $\tilde{\Theta} = \frac{1}{K} \sum_{k\in[0,K-1]} \Theta_k$
            4. Return $\tilde{\Theta}$
    """
    BootstrapAssert(t_params)

    # 2. & 3.
    Theta_k = [None]*t_params.num_subdatasets
    for k in range(t_params.num_subdatasets):
        Theta_k[k] = est_k(t_data, t_params, t_obs, **obs_kwargs)

    # 4. & 5. & 6.
    return skeleton_est(Theta_k,t_params)

def var(t_data, t_params, t_obs = np.average, **obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: BootstrapParams
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data. This can be useful if
            the t_obs outputs only the required estimators in each subdata set.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Estimator

        This function uses the bootstrap method to determine the estimator of the
        observable determined over t_data.
        Let $N$ be the size of the data set. Let $X = {x_i}_{i\in[N]}$ denote
        the total data set and K be the number of subdata_sets
            1. On data X perform bootstrap            --> ${X_k}_{k\in[0,K-1]}$
            2. Compute observable on each subdata set --> $\Theta_k=\Theta(X_k)$
            3. Compute estimator                      --> $\tilde{\Theta} = \frac{1}{K} \sum_{k\in[0,K-1]} \Theta_k$
            4. Return $\tilde{\Theta}$
    """
    BootstrapAssert(t_params)

    # 2. & 3.
    Theta_k = [None]*t_params.num_subdatasets
    for k in range(t_params.num_subdatasets):
        Theta_k[k] = est_k(t_data, t_params, t_obs, **obs_kwargs)

    # 4. & 5. & 6.
    return skeleton_var(Theta_k,t_params)

# ==============================================================================
# Ready to go Bootstrap method computing estimator and variance
# ==============================================================================

def bootstrap(t_data, t_params, t_obs = np.average, **obs_kwargs):
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
    # define return values
    est = None
    var = None

    if t_params.with_blocking:
        from ..Blocking import get_block
        from ..analysisParams import BlockingParams

        bl_params = BlockingParams(t_params.data_size,t_params.num_blocks)

        l_est = [None]*bl_params.num_blocks
        l_var = [None]*bl_params.num_blocks

        for block_id in range(bl_params.num_blocks):
            data_block = get_block(t_data,block_id,bl_params)

            Theta_k = [None]*t_params.num_subdatasets
            for k in range(t_params.num_subdatasets):
                Theta_k[k] = est_k(data_block, t_params, t_obs, **obs_kwargs)

            l_est[block_id] = skeleton_est(Theta_k,t_params)
            l_var[block_id] = skeleton_var(Theta_k,t_params)

        est = np.average(l_est,axis=0)
        var = np.average(l_var,axis=0)
    else:
        # 2. & 3.
        Theta_k = [None]*t_params.num_subdatasets
        for k in range(t_params.num_subdatasets):
            Theta_k[k] = est_k(t_data, t_params, t_obs, **obs_kwargs)

        # 4. & 5.
        est = skeleton_est(Theta_k,t_params)
        var = skeleton_var(Theta_k,t_params)

    return est,var
