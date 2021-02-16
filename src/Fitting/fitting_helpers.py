import numpy as np
import itertools

def res(A):
    return np.linalg.norm( A - np.identity(A.shape[0]) )

def cov(t_data,t_analysis_params=None):
    r"""
        t_data: numpy.ndarray
            Represents the data over which the covariance matrix has to be determined
            Require 2 dimensions e.g. shape = (#gauge,#Nt)
        t_analysis_params: AnalysisParams, default: None
            Specification of the analysis method. If set to `None` the covariance
            matrix is estimated using np.cov. Else it is estimated using the
            specified analysis method. e.g. bootstrap
    """
    def _cov(t_data):
        return np.cov(t_data.T)

    if t_analysis_params is None:
        return _cov(t_data)
    else:
        if t_analysis_params.analysis_type == "jackknife":
            from ..analysis.Jackknife import est
        elif t_analysis_params.analysis_type == "bootstrap":
            from ..analysis.Bootstrap import est
        elif t_analysis_params.analysis_type == "blocking":
            from ..analysis.Blocking import est

        return est(t_data, t_analysis_params, t_obs = _cov)

def cor(t_data,t_analysis_params=None):
    r"""
        t_data: numpy.ndarray
            Represents the data over which the covariance matrix has to be determined
            Require 2 dimensions e.g. shape = (#gauge,#Nt)
        t_analysis_params: AnalysisParams, default: None
            Specification of the analysis method. If set to `None` the covariance
            matrix is estimated using np.cov. Else it is estimated using the
            specified analysis method. e.g. bootstrap
    """
    def _cor(t_data):
        return np.corrcoef(t_data.T)

    if t_analysis_params is None:
        return _cor(t_data)
    else:
        if t_analysis_params.analysis_type == "jackknife":
            from ..analysis.Jackknife import est
        elif t_analysis_params.analysis_type == "bootstrap":
            from ..analysis.Bootstrap import est
        elif t_analysis_params.analysis_type == "blocking":
            from ..analysis.Blocking import est

        return est(t_data, t_analysis_params, t_obs = _cor)

def cov_fit_param(t_abscissa,t_ordinate,t_cov_inv,t_model,t_params,t_inv_acc=1e-8):
    r"""
        t_abscissa: numpy.array
            Axis above which the model becomes evaluated
        t_ordinate: numpy.array
            Axis against which the model is fitted
        t_cov_inv: numpy.ndarray
            Inverse covariance matrix of the fit (can be diagonal) must be two dimensional
        t_model: qcdanalysistools.fitting.ModelBase
            Model which is fitted as defined in src/Fitting/model.py
        t_params:
            Best fit params to evaluate the models jacobian
        t_inv_acc: float, default: 1e-8
            Accuracy which is accepted for inverting the computed inverse covariance

        This function implements the computation of the covariance of the
        fit parameters i.e.
        $$
            Cov[\Theta] = \left( J_\Theta C^{-1} J_\Theta^T \right)^{-1}
            $$
            where J is the jacobian vector of the model in respect to the parameters $\Theta$
    """
    # compute the parameter jacobian from the model i.e. df(x_i,Theta)/dTheta_a
    # The shape must be (num_params,t_abscissa.size)
    J = t_model.grad_param(t_abscissa,*t_params)

    if J.shape != (t_model.num_params,t_abscissa.size):
        raise ValueError(f"Jacobian of model {t_model.__name__()} has wrong shape: {J.shape}. Require a shape of ({(t_model.num_params,num_axis_points)})")

    # allocate memory for the inverse covariance matrix of dimension #params x #params
    cov_inv = np.zeros(shape=(t_model.num_params,t_model.num_params))

    for a,b in itertools.product(range(t_model.num_params),repeat=2):
        cov_inv[a,b] = np.dot( np.dot( np.transpose(J[a,:]), t_cov_inv ) , J[b,:] )

    # invert the expression
    try:
        cov = np.linalg.inv(cov_inv)
    except:
        print(f"WARNING: Require SVD to invert parameter covariance matrix.")
        u,w,v = np.linalg.svd(cov_inv)
        cov = np.dot(np.dot(np.transpose(v),np.diag(np.divide(np.ones(w.size),w,out=np.zeros(w.size),where=w<t_inv_acc**2))),np.transpose(u))

    # check that the inverse worked out
    res_r = res(cov @ cov_inv)
    res_l = res(cov_inv @ cov)

    if res_r > t_inv_acc:
        raise ValueError(f"Matrix right inverse of the fit parameter covariance matrix is not precise: residuum = {res_r}")
    if res_l > t_inv_acc:
        raise ValueError(f"Matrix left inverse of the fit parameter covariance matrix is not precise: residuum = {res_l}")

    return cov

def cov_fit_param_est(t_param_data,t_analysis_params):
    r"""
        t_param_data: np.ndarray
            Parameters obtained from each sample.
            It is assumed that
    """

    def cov_est(t_param_data):
        cov = np.zeros(shape=(t_param_data.shape[1],t_param_data.shape[1]))

        for i,j in itertools.product(range(t_param_data.shape[1]),repeat=2):
            cov[i,j] = np.average( t_param_data[:,i] * t_param_data[:,j] ) \
                     - np.average( t_param_data[:,i])* np.average( t_param_data[:,j])

        return cov

    if t_analysis_params is None:
        return cov_est(t_param_data)

    if t_analysis_params.analysis_type == "jackknife":
        from ..analysis.Jackknife import est
    elif t_analysis_params.analysis_type == "bootstrap":
        from ..analysis.Bootstrap import est
    elif t_analysis_params.analysis_type == "blocking":
        from ..analysis.Blocking import est

    return est(t_data = t_param_data , t_params = t_analysis_params, t_obs = cov_est)
