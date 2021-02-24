
r"""
    This file contains a base fitting class which is used in all subsequent fitting methods.
"""
import numpy as np
import scipy.optimize as opt
import itertools
import warnings

from ..analysis import estimator

class FitBase():
    def __init__(self,t_model,t_abscissa,t_data=None,t_ordinate=None,t_analysis_params=None):
        r"""
            t_model: qcdanalysistools.fitting.model
                A model to which the data should be fit. Commonly, it needs to
                implement a function
                    * t_mode.hess_param(x,*Theta):
                        Computing the hessian of the model function
                        in respect to the parameters thus needs to
                        return an array of size (num_params,num_params)
                    * t_mode.grad_param(x,*Theta):
                        Computing the jacobian of the model
                        function in respect to the parameters
                        thus needs to return an array of size
                        (num_params,).
                    * t_mode.__call__(x,*Theta):
                        Computing the model function at a given input x
                        and parameters *Theta = (Theta_0,Theta_1,...)
                    * num_params:
                        Number of parameters to fit to
                    * Theta0:
                        Tuple of first guess for each parameter.
            t_abscissa: numpy.array
                Abscissa used to evaluate the model function. Needs to be of
                shape (D,).
            t_data: numpy.ndarray
                Results from a lattice qcd simulation to which the model should
                be fitted. Needs to be of shape (N,D) where N is then number of
                configurations. Can be None but then t_ordinate must be given!
            t_ordinate: numpy.array
                If the data is already processed (e.g. averaged) the ordinate to
                which the model is fitted can be given explicitly. This is required
                if t_data is None.
            t_analysis_params: qcdanalysistools.analysis.AnalysisParams
                Is one of the analysis parameter instantation defined in
                    src/StatisticalAnalysis/analysisParams.py
                Used to preprocess the data but can be None. Then preprocessing
                is achived with numpy.average.

            This class represents the base class to all fitting methods.
            It handles the basic data preparation i.e. preparing the ordinate and
            abscissa for the fit. (Co)variance handling has to be done in derived
            classes at it is not general that any method will use those.

            Notes:

            * All classes deriving from this must implement
                * self.fit()
                    This method is used in the __call__ implementation.
                * self.print_result()

        """
        # store the model: models are defined in model.py or selfdeveloped based
        # on qcdanalysistools.fitting.ModelBase
        self.model = t_model
        # store the analysis parameter
        self.analysis_params = t_analysis_params
        # Store the desired frequencies i.e. the data to fit against
        # The classes will manage the data and generate a ordninate from it if not given.
        if t_data is None:
            # If t_data is not given the ordninate needs to be given and set here
            # the ordinate_cov/ordinate_var must be apperent. This has to be
            # computed in the derived classes.
            if t_ordinate is None:
                raise ValueError(f"Fitting requires either t_data or t_ordinate")
            if len(t_ordinate.shape) != 1:
                raise ValueError(f"t_ordinate must be 1-dimensional but is of shape {t_ordinate.shape}")

            self.ordinate = t_ordinate
        else:
            # If t_data is apperent but the ordinate ist not store the data and
            # compute the ordinate with the given analysis type

            # data must be 2-dimensional e.g. (#gauge,D=Nt...)
            if len(t_data.shape) != 2:
                raise ValueError(f"t_data must be 2-dimensional but is of shape {t_ydata.shape}")

            # store the data
            self.data = t_data

            if t_ordinate is None:
                if t_analysis_params is None: # ToDo: This should be available within ..analysis
                    # fallback to standard average if no analysis method is given
                    self.ordinate = np.average(self.data,axis=0)
                self.ordinate = estimator(t_analysis_params,self.data)
            else:
                # if the data is given and also the ordinate, store it
                if len(t_ordinate.shape) != 1:
                    raise ValueError(f"t_ordinate must be 1-dimensional but is of shape {t_ordinate.shape}")

                self.ordinate = t_ordinate

        # Store the argument of the model e.g. time axis.
        # it must have same dimension as the computed or given ordinate
        self.abscissa = t_abscissa

        # Check that ordinate and abscissa extend match
        if self.abscissa.size != self.ordinate.size:
            raise ValueError(f"Abscissa size ({self.abscissa.size}) and ordinate size ({self.ordinate.size}) must match!")

        # store the report from the minimization procedure i.e. scipy.optimize.OptimizeResult
        self.min_stats = dict()

        # store the fit report from the fitting procedure i.e. dictionary filled
        # once the self.fit() method is called
        self.fit_stats = dict()

        # list of all solvers within the scipy.optimize library. All are required
        # WARNING: DO NOT CHANGE THIS LIST UNLESS YOU KNOW WHAT YOU ARE DOING!!!
        self.list_of_minimize_methods = ['Nelder-Mead',  #
                                         'Powell',       #
                                         'COBYLA',       #
                                         'CG',           # jacobian
                                         'BFGS',         # jacobian
                                         'L-BFGS-B',     # jacobian
                                         'TNC',          # jacobian
                                         'SLSQP',        # jacobian
                                         'Newton-CG',    # jacobian, hessian
                                         'dogleg',       # jacobian, hessian
                                         'trust-constr', # jacobian, hessian
                                         'trust-ncg',    # jacobian, hessian
                                         'trust-exact',  # jacobian, hessian
                                         'trust-krylov', # jacobian, hessian
                                        ]

    def chisq(self,params):
        raise NotImplementedError

    def grad_chisq(self,params):
        raise NotImplementedError

    def hess_chisq(self,params):
        raise NotImplementedError

    def _fit(self):
        r"""
            Calls the different minimization methods of scipy.optimize to minimize
            the objective function self.chisq
            Requires implementation of:
                * self.chisq
                * self.grad_chisq
                * self.hess_chisq
            Bases on the order of self.list_of_minimize_methods
        """
        min_res_list = []

        # scipy throws overflow warnings which should not be printed here as those minimizations have no success they become removed anyway
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i_method, method in enumerate(self.list_of_minimize_methods):
                #print(f"Correlated Least Square: Trying minimization of \u1d61\u00B2 with {method}", end='... ')
                # 2. minimize, note not all use gradient and hessian information
                #              the if condition take care of it. This is why the order
                #              of list_of_minimize_methods is important
                # WARNING: DO NOT CHANGE THESE IF CONDITION UNLESS YOU KNOW WHAT YOU ARE DOING!!!
                if i_method < 3:
                    # they do not use jacobian nore hessp
                    min_res = opt.minimize(self.chisq,self.model.params0,method=method)
                elif 2 < i_method < 8:
                    # they do not use hessp
                    min_res = opt.minimize(self.chisq,self.model.params0,method=method,jac=self.grad_chisq)
                else:
                    # they use both
                    min_res = opt.minimize(self.chisq,self.model.params0,method=method,jac=self.grad_chisq,hess=self.hess_chisq)

                # 3. only append results if minimization succeeded
                if min_res['success']:
                    min_res_list.append(min_res)
                    #print("succeeded")
                #else:
                    # I think it is not important why minimization did not work. If you need to know interchange these print statements.
                    #print(f"failed:\n {min_res}")
                    #print("failed")

        return min_res_list

    def fit(self,*args,**kwargs):
        raise NotImplementedError

    def print_result(self):
        raise NotImplementedError

    def __call__(self,*args,**kwargs):
        return self.fit(*args,**kwargs)
