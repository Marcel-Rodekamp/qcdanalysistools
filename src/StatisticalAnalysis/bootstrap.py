r"""
    This file contains the
        * Bootstrap
    method.
"""
import numpy as np
from .jackknife import _leave_n_out_ran

def _leave_out(t_data,t_num_subdata_sets):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_num_subdata_sets: int
            List of random indices which are left out.

        Returns:
            A set of subdata where each subdata is the original data set but without
            one randomly chosen data point
    """
    # The bootstrap methods leaves 1 randomly chosen data point out.
    # Following function can be recycled for this pupose. Therefore, the
    # default case t_n = 1 is used.
    return _leave_n_out_ran(t_data,t_num_subdata_sets,t_n=1)

def bootstrap_est(t_data,t_num_subdata_sets = None,t_obs=np.average,**obs_kwargs):
    """
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_num_ran_indices: int, default: None
            Number of random datasets created by leave one random out. Defaults
            to t_data.shape[0] // 2 if None is given.
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Estimator

        This function uses the bootstrap method to determine the estimator of the
        observable determined over t_data.
        Let $N$ be the size of the data set. Let $X = {x_i}_{i\in[N]}$ denote
        the total data set and K be the number of subdata_sets
            1. On data X perform leave one random out --> ${X_k}_{k\in[0,K-1]}$
            2. Compute observable on each subdata set --> $\Theta_k=\Theta(X_k)$
            3. Compute estimator                      --> $\tilde{\Theta} = \frac{1}{K} \sum_{k\in[0,K-1]} \Theta_k$
            4. Return $\tilde{\Theta}$
    """
    # 1. create subdata sets
    t_num_subdata_sets = t_data.shape[0]//2 if t_num_subdata_sets==None else t_num_subdata_sets
    subdata_sets = _leave_out(t_data,t_num_subdata_sets=t_num_subdata_sets)

    # 2. Compute observables
    Theta_k = [None]*t_num_subdata_sets
    for k,x_k in enumerate(subdata_sets):
        Theta_k[k] = t_obs(x_k,**obs_kwargs)

    # 3. determine estimator
    return np.average(np.array(Theta_k), axis = 0)

def bootstrap_var(t_data,t_num_subdata_sets = None,t_obs=np.average,**obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_num_subdata_sets: int, default: None
            Number of random datasets created by leave one random out. Defaults
            to t_data.shape[0] // 2 if None is given.
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Variance

        This function uses the bootstrap method to determine the estimator of the
        observable determined over t_data.
        Let $N$ be the size of the data set. Let $X = {x_i}_{i\in[N]}$ denote
        the total data set and K be the number of subdata sets
            1. On data X perform leave one random out --> ${X_k}_{k\in[0,N-1]}$
            2. Compute observable on each subdata set --> $\Theta_k=\Theta(X_k)$
            3. Compute estimator                      --> $\tilde{Theta} = \frac{1}{K}\sum_{k\in[0,K-1]} \Theta_k$
            4. Compute variance                       --> $\sigma^2 = \frac{1}{K} \sum_{k\in[0,K-1]} \left(\Theta_k-\tilde{\Theta}\right)^2$
    """
    # 1. create subdata sets
    t_num_subdata_sets = t_data.shape[0]//2 if t_num_subdata_sets==None else t_num_subdata_sets
    subdata_sets = _leave_out(t_data,t_num_subdata_sets=t_num_subdata_sets)

    # 2. Compute observables
    Theta_k = [None]*t_num_subdata_sets
    for k,x_k in enumerate(subdata_sets):
        Theta_k[k] = t_obs(x_k,**obs_kwargs)

    # 3,4. determine variance
    return np.var(np.array(Theta_k), axis = 0)

def bootstrap(t_data,t_num_subdata_sets = None, t_obs = np.average, t_blocked = False, t_num_blocks = None, **obs_kwargs):
    r"""
        t_data: numpy.ndarray
            Data array, containing the raw data. It is assumed,
            that axis=0 represents the data points of the method and subsequent
            axis' are assumed to represent multidimensional estimators.
        t_num_subdata_sets: int, default: None
            Number of random datasets created by leave one random out. Defaults
            to t_data.shape[0] // 2 if None is given.
        t_obs: function, default: numpy.average
            Observable which should be computed over t_data.
        **obs_kwargs: keyworded arguments
            Keyworded arguments passed to t_obs

        Returns: numpy.ndarray
            Estimator,Variance
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
            l_est[block_id] = bootstrap_est(blocking.get_block(t_data,block_id,block_size),t_num_subdata_sets=t_num_subdata_sets,t_obs=t_obs,**obs_kwargs)
            l_var[block_id] = bootstrap_var(blocking.get_block(t_data,block_id,block_size),t_num_subdata_sets=t_num_subdata_sets,t_obs=t_obs,**obs_kwargs)

        l_est[t_num_blocks-1] = bootstrap_est(blocking.get_block(t_data,t_num_blocks-1,block_size),t_num_subdata_sets=t_num_subdata_sets,t_obs=t_obs,**obs_kwargs)
        l_var[t_num_blocks-1] = bootstrap_var(blocking.get_block(t_data,t_num_blocks-1,block_size),t_num_subdata_sets=t_num_subdata_sets,t_obs=t_obs,**obs_kwargs)

        est = np.average(l_est,axis=0)
        var = np.average(l_var,axis=0)
    else:
        # simple leave n out (randomnized)
        est = bootstrap_est(t_data,t_num_subdata_sets=t_num_subdata_sets,t_obs=t_obs,**obs_kwargs)
        var = bootstrap_var(t_data,t_num_subdata_sets=t_num_subdata_sets,t_obs=t_obs,**obs_kwargs)

    return est,var
