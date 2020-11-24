r"""
    This file contains the
        * leave n out jackknife
        * leave n random out jackknife
    methods.

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
    import qcdanalysistools.analysis.JackknifeParams
    from qcdanalysistools.analysis.Jackknife import *

    full_data = ...

    # 1. Create parameters to specify jackknife
    jkk_params = JackknifeParams(t_data_size,t_n,t_random_leaveout,t_num_ran_indices)

    # 2. Define observable/estimator function acting on one subdata set
    def obs(t_kth_subdataset,*args,**kwargs):
        ...

    # 3. Compute Estimator on full data set yourself
    # Note if blocking is required this mus be implemented here aswell!
    full_obs_out = obs(full_data,*args,**kwargs)
    Theta_hat = full_obs_out.estimated_quantity # or however one accesses it

    # 4. Compute the estimators for each subdata set
    obs_out = [None]*jkk_params.num_subdatasets
    Theta_k = np.zeros(shape = (jkk_params.num_subdatasets,...))

    for k in range(jkk_params.num_subdatasets):
        obs_out[k] = obs(subdataset(full_data,k,jkk_params),*args,**kwargs)
        Theta_k[k] = obs_out[k].estimated_quantity # or however one accesses it

    # 5. Compute the biased improved estimator and variance
    est = skeleton_est(Theta_k,Theta_hat,jkk_params)
    var = skeleton_var(Theta_k,Theta_hat,jkk_params)

    # 6. Output
    ...

    ```
"""
import numpy as np
from ..analysisParams import JackknifeParams

# ==============================================================================
# Assertion based on the parameter
# ==============================================================================
def JackknifeAssert(t_params):
    if not isinstance(t_params,JackknifeParams):
        raise RuntimeError(f"Jackknife does not work with parameter of type {t_params.analysis_type}.")

# ==============================================================================
# Creating Data Subsets following the Jackknife, leave n out, methods
# ==============================================================================

def subdataset(t_data, t_k, t_params):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_k: int
            Index of the subdata set
        t_params: JackknifeParams
            Parameter of the jackknife method

        Returns: numpy.ndarray
            Returns the kth subdata set containing N-t_n points that is the
            total size of t_data minus the amount of left out data points.
    """
    JackknifeAssert(t_params)
    if t_k >= t_data.shape[0]//t_params.n:
        raise ValueError(f"t_k ({t_k}) to large for data set t_data of size ({t_data.shape[0]}) with n ({t_params.n})")

    # delete t_n from the full data block and store it in the subdata matrix
    if t_params.random_leaveout:
        return np.delete(t_data,[t_params.leave_out_index_list[t_k]+i if t_params.leave_out_index_list[t_k]+i<t_params.data_size else i for i in range(t_params.n)],axis=0)
    else:
        return np.delete(t_data,[t_k+i if t_k+i<t_params.data_size else i for i in range(t_params.n)],axis=0)

def subdatasets(t_data, t_params):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_params: JackknifeParams
            Parameter of the jackknife method

        Returns: numpy.ndarray
            Returns the set if subdata sets containing N-t_n points that is the
            total size of t_data minus the amount of left out data points.
    """
    JackknifeAssert(t_params)
    # initialize tensor over subdata sets
    subdata_sets = np.zeros( shape=( t_params.num_subdatasets, t_params.data_size-t_n, *t_data.shape[1:] ) )

    for k in range(t_params.num_subdatasets):
        subdata_sets[k] = subdataset(t_data,k,t_params)

    return subdata_sets

# ==============================================================================
# Bits and Pieces
# ==============================================================================

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
    JackknifeAssert(t_params)
    return t_obs(subdataset(t_data, t_k, t_params),**obs_kwargs)

def skeleton_est(t_estimators,t_biased_est,t_params):
    r"""
        t_estimators: numpy.ndarray
            Estimators computed on each subdata set in the jackknife method.
        t_biased_est: numpy.ndarray or float
            Biased estimator coming from a estimation on the full data set.
        t_params: JackknifeParams
    """
    JackknifeAssert(t_params)

    if not isinstance(t_estimators, np.ndarray):
        try:
            t_estimators = np.array(t_estimators)
        except:
            raise ValueError(f"The estimators need to be of list type but are ({type(t_estimators)}).")


    if t_estimators.shape[0] != t_params.num_subdatasets:
        raise ValueError(f"The estimators on subdata sets are not valid, require axis 0 of length ({t_params.num_subdatasets}) but got (t_estimators.shape[0]).")

    Theta_tilde = np.average(t_estimators,axis=0)

    return t_params.num_subdatasets*t_biased_est-(t_params.num_subdatasets-1)*Theta_tilde

def skeleton_var(t_estimators,t_biased_est,t_params):
    r"""
        t_estimators: numpy.ndarray
            Estimators computed on each subdata set in the jackknife method.
        t_biased_est: numpy.ndarray or float
            Biased estimator coming from a estimation on the full data set.
        t_params: JackknifeParams
    """
    JackknifeAssert(t_params)

    if not isinstance(t_estimators, np.ndarray):
        try:
            t_estimators = np.array(t_estimators)
        except:
            raise ValueError(f"The estimators need to be of list type but are ({type(t_estimators)}).")


    if t_estimators.shape[0] != t_params.num_subdatasets:
        raise ValueError(f"The estimators on subdata sets are not valid, require axis 0 of length ({t_params.num_subdatasets}) but got (t_estimators.shape[0]).")

    return ((t_params.num_subdatasets-1)/t_params.num_subdatasets)*np.sum(np.square(t_estimators-t_biased_est),axis=0)

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
    JackknifeAssert(t_params)
    # 1.
    Theta_hat = t_obs(t_data,**obs_kwargs)

    # 2. & 3.
    Theta_k = [None]*t_params.num_subdatasets
    for k in range(t_params.num_subdatasets):
        Theta_k[k] = est_k(t_data, t_params, k, t_obs, **obs_kwargs)

    # 4. & 5. & 6.
    return skeleton_est(Theta_k,Theta_hat,t_params)

# ==============================================================================
# Ready to go variance
# ==============================================================================

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

        This functions uses the leave n out jackknife method in order to determine
        the estimator of the observable determined over t_data.
        Let $N$ be the size of the data set and $K = \frac{N}{t_n}$ be the number
        of subdata sets. Let $X = {x_i}_{i\in[N]}$ denote the total data set
            1. Compute Observable on Dataset          --> $\hat{\Theta}=\Theta(X)$
            2. On data X perform a leave n out method --> ${X_k}_{k\in[0,K-1]}$
            3. Compute observable on each subdata set --> $\Theta_k=\Theta(X_k)$
            4. Compute variance                       --> $\sigma^2 = \frac{K-1}{K} \sum_{k\in[0,K-1]} \left( \Theta_k - \hat{\Theta} \right)^2$
            5. Return $\sigma^2$
    """

    # 1. determine the estimator on the full data set
    Theta_hat = t_obs(t_data, **obs_kwargs)

    # 2. & 3.
    Theta_k = [None]*t_params.num_subdatasets
    for k in range(t_params.num_subdatasets):
        Theta_k[k] = est_k(t_data, t_params, k, t_obs, **obs_kwargs)

    # 4. & 5.
    return skeleton_var(Theta_k,Theta_hat,t_params)

# ==============================================================================
# Ready to go Jackknife method computing estimator and variance
# ==============================================================================

def jackknife(t_data, t_params, t_obs = np.average, **obs_kwargs):
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

        bl_params = BlockingParams(t_data.shape[0],t_params.num_blocks)

        l_est = [None]*bl_params.num_blocks
        l_var = [None]*bl_params.num_blocks

        for block_id in range(bl_params.num_blocks):
            data_block = get_block(t_data,block_id,bl_params)

            Theta_hat = t_obs(data_block,**obs_kwargs)

            Theta_k = [None]*t_params.num_subdatasets
            for k in range(t_params.num_subdatasets):
                Theta_k[k] = est_k(data_block, t_params, k, t_obs, **obs_kwargs)

            l_est[block_id] = skeleton_est(Theta_k,Theta_hat,t_params)
            l_var[block_id] = skeleton_var(Theta_k,Theta_hat,t_params)

        est = np.average(l_est,axis=0)
        var = np.average(l_var,axis=0)
    else:
        # 1.
        Theta_hat = t_obs(t_data,**obs_kwargs)

        # 2. & 3.
        Theta_k = [None]*t_params.num_subdatasets
        for k in range(t_params.num_subdatasets):
            Theta_k[k] = est_k(t_data, t_params, k, t_obs, **obs_kwargs)

        # 4. & 5.
        est = skeleton_est(Theta_k,Theta_hat,t_params)
        var = skeleton_var(Theta_k,Theta_hat,t_params)

    return est,var
