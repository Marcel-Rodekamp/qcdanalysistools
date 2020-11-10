r"""
    This file contains classes for least square fitting
        * LeastSquare: Non correlated data i.e. Cov[ydata] is diagonal
        * LeastSquareCorr: Correlated data i.e. Cov[ydata] is fully populated

    ToDo:
        * Implement non linear error estimation i.e. get covariance from
          bootstrap over real data
        * abstract the fit result output into an own class which automatically
          determines required statistics s.t.
            * x²
            * bayschen
            * covarianc
            * error
"""

import numpy as np
import itertools
# Extensively use this function to find a optimal minimization of a cost function
from scipy.optimize import minimize
# import base class
from .fitting_base import FitBase

class LeastSquare(FitBase):
    def __init__(self,t_model,t_ydata,t_xdata,*min_args,**min_kwargs):
        r"""
            t_model: qcdanalysistools.fitting.model
                A model to which the data should be fit. Commonly, it needs to
                implement a function
                    * t_mode.jac_param(x):
                        Computing the jacobian of the model
                        function in respect to the parameters
                        thus needs to return an array of size
                        #parameters.
                    * t_mode.apply(x,*Theta):
                        Computing the model function at a given input x and parameters
                        *Theta = (Theta_0,Theta_1,...)
                    * num_params:
                        Number of parameters to fit to
                    * t_Theta0:
                        Tuple of first guess for each parameter.
            t_ydata: numpy.ndarray
                Array representing the data to which the model is fit s.t.
                    ```
                        t_ydata - t_model(x,*Theta)
                    ```
                becomes minimized.
            t_xdata: numpy.ndarray
                Array representing the arguments of the of the fit model.
                The model is evaluated at these points to minimize a residual.
            *min_args: arguments
                Optional arguments passed to `scipy.optimize.minimize`
            **min_kwargs: keyworded arguments
                Optional keyworded arguments passed to `scipy.optimize.minimize`
        """
        # initialize base class
        super().__init__(t_model,t_ydata,t_xdata,*min_args,**min_kwargs)

        # precompute the variances
        self.y_var = np.var(self.ydata, axis = 0)

        # compute y data estimator
        self.y_avg = np.average(self.ydata,axis=0)

    def fit(self,*args,**kwargs):
        r"""
            *args: arguments
                Arguments passed to the model which are not parameters to fit
            *kwargs: keyworded arguments
                Keyworde arguments passed to the model which are not parameters to fit
        """
        # define chisq which is going to be minimized over parameters Theta
        chisq = lambda Theta : np.sum(np.square(self.y_avg-self.model.apply(self.xdata,*Theta,*args,**kwargs))/(2*self.y_var))

        # minimize chisq over
        self.min_stats = minimize(chisq, self.model.Theta0, *self.min_args, **self.min_kwargs)
        # store the best fit parameter
        self.fit_stats['Param'] = self.min_stats['x']
        # store best fit data points evaluated over xdata
        self.fit_stats['Best fit'] = self.model.apply(self.xdata,*self.min_stats['x'])
        # compute and store the covariance matrix of the fit
        self.fit_stats['Cov'] = self._cov()
        # compute and store the fit error
        self.fit_stats['Fit error'] = np.sqrt(np.diag(self.fit_stats['Cov']))

        # return the report
        return self.fit_stats

class LeastSquareCorr(FitBase):
    def __init__(self,t_model,t_ydata,t_xdata,*min_args,**min_kwargs):
        r"""
            t_model: qcdanalysistools.fitting.model
                A model to which the data should be fit. Commonly, it needs to
                implement a function
                    * t_mode.jac_param(x):
                        Computing the jacobian of the model
                        function in respect to the parameters
                        thus needs to return an array of size
                        #parameters.
                    * t_mode.apply(x,*Theta):
                        Computing the model function at a given input x and parameters
                        *Theta = (Theta_0,Theta_1,...)
                    * num_params:
                        Number of parameters to fit to
                    * t_Theta0:
                        Tuple of first guess for each parameter.
            t_ydata: numpy.ndarray
                Array representing the data to which the model is fit s.t.
                    ```
                        t_ydata - t_model(x,*Theta)
                    ```
                becomes minimized.
            t_xdata: numpy.ndarray
                Array representing the arguments of the of the fit model.
                The model is evaluated at these points to minimize a residual.
            *min_args: arguments
                Optional arguments passed to `scipy.optimize.minimize`
            **min_kwargs: keyworded arguments
                Optional keyworded arguments passed to `scipy.optimize.minimize`
        """
        # initialize base class
        super().__init__(t_model,t_ydata,t_xdata,*min_args,**min_kwargs)

        # freeze correlation matrix, this is preferred over covariance as it is
        # better conditioned for the matrix inverse
        self.cor_y = np.corrcoef(self.ydata.T)

        # compute its inverse
        self.cor_y_inv = np.linalg.inv(self.cov_y)

        # precompute the variances
        self.y_var = np.var(self.ydata, axis = 0)

        # compute y data estimator
        self.y_avg = np.average(self.ydata,axis=0)


    def _chisq(self,t_Theta,*args,**kwargs):
        r"""
            t_Theta: tuple
                Tuple of parameters passed to the model and over which the
                chisq is minimized.
            *args: arguments
                Arguments passed to the model which are not parameters to fit
            *kwargs: keyworded arguments
                Keyworde arguments passed to the model which are not parameters to fit

            LeastSquare function applying the approximate correlation function
            determined over the raw data set to account for correlations.
        """
        # e.g. number of time slices Nt
        num_axis_points = self.ydata.shape[1]

        xsq = 0

        for t1,t2 in itertools.product(range(num_axis_points),repeat=2):
            xsq += (self.y_avg[t1]-self.model.apply(self.xdata[t1],*t_Theta,*args,**kwargs))/self.y_var[t1] \
            * self.cor_y_inv[t1,t2] * \
            (self.y_avg[t2]-self.model.apply(self.xdata[t2],*t_Theta,*args,**kwargs))/self.y_var[t2]

        return xsq

    def fit(self,*args,**kwargs):
        r"""
            *args: arguments
                Arguments passed to the model which are not parameters to fit
            *kwargs: keyworded arguments
                Keyworde arguments passed to the model which are not parameters to fit
        """
        # define chisq which is going to be minimized over parameters Theta
        chisq = lambda Theta : self._chisq(Theta,*args,**kwargs)

        # minimize chisq over
        self.min_stats = minimize(chisq, self.model.Theta0, *self.min_args, **self.min_kwargs)
        # store the best fit parameter
        self.fit_stats['Param'] = self.min_stats['x']
        # store best fit data points evaluated over xdata
        self.fit_stats['Best fit'] = self.model.apply(self.xdata,*self.min_stats['x'])
        # compute and store the covariance matrix of the fit
        self.fit_stats['Cov'] = self._cov()
        # compute and store the fit error
        self.fit_stats['Fit error'] = np.sqrt(np.diag(self.fit_stats['Cov']))

        # return the report
        return self.fit_stats
