r"""
    This file contains classes for least square fitting
        * Uncorrelated Fit (Diagonal approximation) : DiagonalLeastSquare
        * Correlated Fit                            : CorrelatedLeastSquare
"""
import numpy as np
import scipy.optimize as opt
import scipy.stats
import itertools
from ..analysis import estimator,variance,get_sample,checkAnalysisType,Jackknife,Blocking,Bootstrap
from .fitting_base import FitBase
from .fitting_helpers import * # cov,cor,cov_fit_param
from qcdanalysistools.stats import AIC_chisq, AICc_chisq

class DiagonalLeastSquare(FitBase):
    def __init__(self,t_model,t_abscissa,t_data=None,t_ordinate=None,t_ordinate_var=None,t_analysis_params=None):
        r"""
            t_model: qcdanalysistools.fitting.model
                A model to which the data should be fit. Commonly, it needs to
                implement a function
                    * t_mode.hess_param(x):
                        Computing the hessian of the model function
                        in respect to the parameters thus needs to
                        return an array of size (num_params,num_params)
                    * t_mode.grad_param(x):
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
            t_ordinate_var: numpy.array
                If the data is already processed (e.g. variance) the variance can
                be given explicitly. This is required if t_data is None
            t_analysis_params: qcdanalysistools.analysis.AnalysisParams
                Is one of the analysis parameter instantation defined in
                    src/StatisticalAnalysis/analysisParams.py
                Used to preprocess the data but can be None. Then preprocessing
                is achived with numpy.average.
        """
        # initialize the base class giving access to the following class members
        #   self.model
        #   self.analysis_params
        #   self.ordinate
        #   self.data (if t_data is not None)
        #   self.abscissa
        #   self.min_stats
        #   self.fit_stats
        # and class methods
        #   self.fit(self,*args,**kwargs): raise NotImplementedError
        #   self.__call__(self,*args,**kwargs): return self.fit(*args,**kwargs)
        super().__init__(t_model,t_abscissa,t_data=t_data,t_ordinate=t_ordinate,t_analysis_params=t_analysis_params)

        # preprocess for the ordinate variance used in chisq as denominator
        if t_data is None:
            # If t_data is not given the variance of the ordinate needs to be given.
            if t_ordinate_var is None:
                raise ValueError(f"Fitting requires either t_data or t_ordinate_var")
            if len(t_ordinate_var.shape) != 1:
                raise ValueError(f"t_ordinate_var must be 1-dimensional but is of shape {t_ordinate_var.shape}")

            self.data = None

            self.ordinate_var = t_ordinate_var
        else:
            # If t_data is given but the ordinate variance is not determine the
            # variance with the given analysis type
            # Note: checks on data are already done in FitBase
            if t_ordinate_var is None:
                if t_analysis_params is None:
                    # fallback to standard average if no analysis method is given
                    self.ordinate_var = np.var(self.data,axis=0)
                else:
                    self.ordinate_var = variance(t_analysis_params,self.data)
            else:
                # If t_data is given and also variance of the ordinate just store it
                if len(t_ordinate_var.shape) != 1:
                    raise ValueError(f"t_ordinate_var must be 1-dimensional but is of shape {t_ordinate_var.shape}")

                self.ordinate_var = t_ordinate_var

    def chisq(self,params):
        r"""
            params: tuple
                Parameter for which chisq becomes minimized. i.e. fitting params

            Note:
                Least Square chisq:
                    $$
                        \chi^2 = \sum_{t=1}^D \frac{(y_t - f(x_t,\Theta))^2}{\sigma_{y_t}}
                    $$
                where $\Theta$ denotes the vector of parameters
        """
        # evaluate the model
        model_res = self.model(self.abscissa,*params)

        return np.sum( np.square(self.ordinate - model_res)/(2*self.ordinate_var) )

    def grad_chisq(self,params):
        r"""
            params: tuple
                Parameter for which chisq becomes minimized. i.e. fitting params

            Note:
                Least Square chisq gradient:
                    $$
                        \pdv{\chi^2}{\Theta_i} = \sum_{t=1}^D \frac{  \pdv{f(x_t,\Theta)}{\Theta_i} (y_t - f(x_t,\Theta))}{\sigma_{y_t}}
                    $$
                    The ith component is the deriviative in respect to the ith parameter
        """
        # evaluate the model
        model_res = self.model(self.abscissa,*params)
        # evaluate the gradient of the model (i.r.t. the parameters)
        model_grad = self.model.grad_param(self.abscissa,*params)

        # initialize memory for the gradient
        xsq_grad = np.zeros(shape = self.model.num_params)

        # compute the gradient
        for i_param in range(self.model.num_params):
            xsq_grad[i_param] = - np.sum( model_grad[i_param,:]*(self.ordinate-model_res)/self.ordinate_var )

        return xsq_grad

    def hess_chisq(self,params):
        r"""
            params: tuple
                Parameter for which chisq becomes minimized. i.e. fitting params

            Note:
                Least Square chisq gradient:
                    $$
                        \pdv{\chi^2}{\Theta_i}{\Theta_j} = \sum_{t=1}^D \frac{\pdv{f(x_t,\Theta)}{\Theta_i}{\Theta_j}(y_t - f(x_t,\Theta)) - \pdv{f(x_t,\Theta)}{\Theta_i}\pdv{f(x_t,\Theta)}{\Theta_j}}{\sigma_{y_t}}
                    $$
                    The ith component is the deriviative in respect to the ith parameter
        """
        # evaluate model
        model_res = self.model(self.abscissa,*params)
        # evaluate the gradient of the model (i.r.t. the parameters)
        model_grad = self.model.grad_param(self.abscissa,*params)
        # evaluate the hessian of the model (i.r.t. the parameters)
        model_hess = self.model.hess_param(self.abscissa,*params)

        # initialize memory for the hessian
        xsq_hess = np.zeros( shape=(self.model.num_params,self.model.num_params) )

        # compute the hessian
        for i,j in itertools.product(range(self.model.num_params),repeat=2):
            xsq_hess[i,j] = -np.sum( (model_hess[i,j,:]*(self.ordinate-model_res)-model_grad[i,:]*model_grad[j,:])/(self.ordinate_var) )

        return xsq_hess

    def fit(self):
        r"""
            Fitting routine of the diagonal least square method.
            1. For each minimization algorithm in scipy.optimize do
            2.      Minimize chisq.
            3.      if minimize succeeded add result to min_res_list
            4. Choose the best minimization from min_res_list
            5. Return fit results+statistics
        """
        # 1.&2 for each minimization method minimize chisq
        min_res_list = self._fit()

        # Fail if no minimization method succeeded
        if len(min_res_list) == 0:
            raise RuntimeError(f"No minimization technique worked for fitting. Try using different start parameters.")

        # 4. find the smallest chisq of all algorithms
        self.min_stats = min_res_list[0]
        fun = min_res_list[0]['fun']
        # TODO: Do we require a faster algorithm here? Try tree structure then.
        for res in min_res_list[1:]:
            if fun > res['fun']:
                fun = res['fun']
                self.min_stats = res

        # 5. Get fit statistics
        # store the best fit parameter
        self.fit_stats['Param'] = self.min_stats['x']
        # store best fit data points evaluated over xdata
        self.fit_stats['Best fit'] = self.model.apply(self.abscissa,*self.min_stats['x'])
        # compute and store the covariance matrix of the fit using implementation of .fitting_helpers
        self.fit_stats['Cov'] = cov_fit_param(self.abscissa,np.diag(np.divide(np.ones_like(self.ordinate_var),self.ordinate_var)),self.model,self.min_stats['x'])
        # compute and store the fit error
        self.fit_stats['Fit error'] = np.sqrt(np.diag(self.fit_stats['Cov']))
        # define the degrees of freedom
        dof = len(self.abscissa)-self.model.num_params
        # compute reduced chisq
        self.fit_stats['red chisq'] = self.chisq(self.fit_stats['Param']) / dof
        # compute p-value
        self.fit_stats['p-value']  = scipy.stats.chi2.sf(self.chisq(self.fit_stats['Param']),dof)
        if self.data is not None:
            # compute Akaike information criterion for normally distributed errors
            self.fit_stats['AIC'] = AIC_chisq(dof, self.fit_stats['red chisq'])
            # compute Akaike information criterion for small data sets
            self.fit_stats['AICc'] = AICc_chisq(dof, self.data.shape[0], self.fit_stats['red chisq'])

        # return the report this is later also accessible from the class
        return self.fit_stats

    def print_result(self,*args,**kwargs):
        out_str = "=======================================\n"
        out_str+= f"Reporting for model {self.model.__name__()}\n"
        out_str+= "=======================================\n"
        out_str+= "========= Best Fit Parameter: =========\n"
        for i_param in range(self.model.num_params):
            out_str+=f"{self.model.param_names[i_param]} = {self.fit_stats['Param'][i_param]:.6e} \u00B1 {self.fit_stats['Fit error'][i_param]: .6e}\n"

        out_str+= "========= Best Fit Covariance: ========\n"
        for i in range(self.model.num_params):
            for j in range(self.model.num_params):
                out_str+=f"{self.fit_stats['Cov'][i,j]: .2e}  "
            out_str+="\n"

        out_str+= "========= Best Fit \u1d61\u00B2: ================\n"
        out_str+= f"\u1d61\u00B2/dof = {self.fit_stats['red chisq']: .6e}\n"

        out_str+= "========= Best Fit p-value: ===========\n"
        out_str+= f"p-value = {self.fit_stats['p-value']: .6e}\n"

        if self.data is not None:
            out_str+= "========= Best Fit Akaike crit: =======\n"
            out_str+= f"AIC  = {self.fit_stats['AIC']: .6e}\n"
            out_str+= f"AICc = {self.fit_stats['AICc']: .6e}\n"

        print(out_str,*args,**kwargs)

class CorrelatedLeastSquare(FitBase):
    def __init__(self,t_model,t_abscissa,t_data=None,t_ordinate=None,t_ordinate_cov=None,t_analysis_params=None, t_inv_acc=1e-8):
        r"""
            t_model: qcdanalysistools.fitting.model
                A model to which the data should be fit. Commonly, it needs to
                implement a function
                    * t_mode.hess_param(x):
                        Computing the hessian of the model function
                        in respect to the parameters thus needs to
                        return an array of size (num_params,num_params)
                    * t_mode.grad_param(x):
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
            t_ordinate_cov: numpy.ndarray
                If the data is already processed (e.g. covariance) the covariance can
                be given explicitly. This is required if t_data is None
            t_analysis_params: qcdanalysistools.analysis.AnalysisParams
                Is one of the analysis parameter instantation defined in
                    src/StatisticalAnalysis/analysisParams.py
                Used to preprocess the data but can be None. Then preprocessing
                is achived with numpy.average.
        """
        # initialize the base class giving access to the following class members
        #   self.model
        #   self.analysis_params
        #   self.ordinate
        #   self.data (if t_data is not None)
        #   self.abscissa
        #   self.min_stats
        #   self.fit_stats
        # and class methods
        #   self.fit(self,*args,**kwargs): raise NotImplementedError
        #   self.__call__(self,*args,**kwargs): return self.fit(*args,**kwargs)
        super().__init__(t_model,t_abscissa,t_data=t_data,t_ordinate=t_ordinate,t_analysis_params=t_analysis_params)

        # preprocess for the ordinate covariance used in chisq as denominator
        if t_data is None:
            # If t_data is not given the covariance of the ordinate needs to be given.
            if t_ordinate_cov is None:
                raise ValueError(f"Fitting requires either t_data or t_ordinate_cov")
            if len(t_ordinate_cov.shape) != 2:
                raise ValueError(f"t_ordinate_cov must be 2-dimensional but is of shape {t_ordinate_cov.shape}")

            self.ordinate_cov = t_ordinate_cov
        else:
            # If t_data is given but the ordinate variance is not, determine the
            # covariance with the given analysis type
            # Note: checks on data are already done in FitBase
            if t_ordinate_cov is None:
                # use the function from qcdanalysistools.fitting.fitting_helpers
                self.ordinate_cov = cov(t_analysis_params,t_data)
            else:
                # If t_data is given and also variance of the ordinate just store it
                if len(t_ordinate_cov.shape) != 2:
                    raise ValueError(f"t_ordinate_cov must be 2-dimensional but is of shape {t_ordinate_cov.shape}")

                self.ordinate_cov = t_ordinate_cov

            if self.data.shape[0] < 10*(self.ordinate.size+1):
                # if this is the case fit might be biased. Compare
                # C. Michael and A. McKerrell
                # Fitting Correlated Hadron Mass Spectrum Data
                # Liverpool Prepint: LTH342, 1994
                # hep-lat/9412087
                print(f"WARNING: To less datapoints N ({self.data.shape[0]}) for a fit to dimension D ({self.ordinate.size})")
                print(f"WARNING: Require N > 10*(D+1) for a reliable fit")

        # invert the covariance matrix
        self.inv_acc = t_inv_acc
        try:
            self.ordinate_cov_inv = np.linalg.inv(self.ordinate_cov)
        except:
            print(f"WARNING: Require SVD to invert covariance matrix.")
            u,w,v = np.linalg.svd(self.ordinate_cov)
            self.ordinate_cov_inv = np.dot(np.dot(np.transpose(v),np.diag(np.divide(np.ones(w.size),w,out=np.zeros(w.size),where=w<self.inv_acc**2))),np.transpose(u))
        # check that the inversion worked
        def res(A):
            return np.linalg.norm(A-np.identity(A.shape[0]))
        # right inverse
        res_r = res(self.ordinate_cov @ self.ordinate_cov_inv)
        res_l = res(self.ordinate_cov_inv @ self.ordinate_cov)
        if res_r > self.inv_acc:
            raise RuntimeError(f"Failed to right invert the covariance matrix: res = {res_r:.4e}")
        if res_l > self.inv_acc:
            raise RuntimeError(f"Failed to left invert the covariance matrix: res = {res_l:.4e}")

        # store the information
        if res_r > res_l:
            self.fit_stats['Cov inv acc'] = res_r
        else:
            self.fit_stats['Cov inv acc'] = res_l

    def chisq(self,params):
        r"""
            params: tuple
                Parameter for which chisq becomes minimized. i.e. fitting params

            Note:
                Least Square chisq:
                    $$
                        \chi^2 = \sum_{t_1,t_2=1}^D (y_{t_1} - f(x_{t_1},\Theta)) CovInv_{t_1,t_2}(y_{t_2} - f(x_{t_2},\Theta))
                    $$
                where $\Theta$ denotes the vector of parameters
        """
        # evaluate the model
        model_res = self.model(self.abscissa,*params)

        # compute chisq
        xsq = 0
        for t1,t2 in itertools.product(range(self.abscissa.size),repeat=2):
            xsq+=(self.ordinate[t1]-model_res[t1])*self.ordinate_cov_inv[t1,t2]*(self.ordinate[t2]-model_res[t2])

        return xsq

    def grad_chisq(self,params):
        r"""
            params: tuple
                Parameter for which chisq becomes minimized. i.e. fitting params

            Note:
                Least Square chisq gradient:
                    $$
                        \pdv{\chi^2}{\Theta_i} = - 2 * \sum_{t_1,t_2=1}^D \pdv{f(x_{t_1},\Theta)}{\Theta_i} * CovInv_{t_1,t_2} * (y_{t_2} - f(x_{t_2},\Theta))
                    $$
                    The ith component is the deriviative in respect to the ith parameter
        """
        # evaluate the model
        model_res = self.model(self.abscissa,*params)
        # evaluate the gradient of the model (i.r.t. the parameters)
        model_grad = self.model.grad_param(self.abscissa,*params)

        # initialize memory for the gradient
        xsq_grad = np.zeros(shape = self.model.num_params)

        # compute the gradient
        for i_param in range(self.model.num_params):
            for t1,t2 in itertools.product(range(self.abscissa.size),repeat=2):
                xsq_grad[i_param] -= 2*model_grad[i_param,t1]*self.ordinate_cov_inv[t1,t2]*(self.ordinate[t2]-model_res[t2])

        return xsq_grad

    def hess_chisq(self,params):
        r"""
            params: tuple
                Parameter for which chisq becomes minimized. i.e. fitting params

            Note:
                Least Square chisq gradient:
                    $$
                        \pdv{\chi^2}{\Theta_i}{\Theta_j} = -2 \sum_{t_1,t_2=1}^D \pdv{f(x_{t_1},\Theta)}{\Theta_i}{\Theta_j}* CovInv_{t_1,t_2} *(y_{t_2} - f(x_{t_2},\Theta))
                                                                               - \pdv{f(x_{t_1},\Theta)}{\Theta_i} * CovInv_{t_1,t_2} * \pdv{f(x_{t_2},\Theta)}{\Theta_j}
                    $$
                    The ith component is the deriviative in respect to the ith parameter
        """
        # evaluate model
        model_res = self.model(self.abscissa,*params)
        # evaluate the gradient of the model (i.r.t. the parameters)
        model_grad = self.model.grad_param(self.abscissa,*params)
        # evaluate the hessian of the model (i.r.t. the parameters)
        model_hess = self.model.hess_param(self.abscissa,*params)

        # initialize memory for the hessian
        xsq_hess = np.zeros( shape=(self.model.num_params,self.model.num_params) )

        # compute the hessian
        for i,j in itertools.product(range(self.model.num_params),repeat=2):
            for t1,t2 in itertools.product(range(self.abscissa.size),repeat=2):
                xsq_hess[i,j] -= 2*(model_hess[i,j,t1]*self.ordinate_cov_inv[t1,t2]*(self.ordinate[t2]-model_res[t2]) \
                                    -model_grad[i,t1]*self.ordinate_cov_inv[t1,t2]*model_grad[j,t2])

        return xsq_hess

    def fit(self):
        r"""
            Fitting routine of the diagonal least square method.
            1. For each minimization algorithm in scipy.optimize do
            2.      Minimize chisq.
            3.      if minimize succeeded add result to min_res_list
            4. Choose the best minimization from min_res_list
            5. Return fit results+statistics
        """
        # 1.&2 for each minimization method minimize chisq
        min_res_list = self._fit()

        # Fail if no minimization method succeeded
        if len(min_res_list) == 0:
            raise RuntimeError(f"No minimization technique worked for fitting. Try using different start parameters.")

        # 4. find the smallest chisq of all algorithms
        self.min_stats = min_res_list[0]
        fun = min_res_list[0]['fun']
        # TODO: Do we require a faster algorithm here? Try tree structure then.
        for res in min_res_list[1:]:
            if fun > res['fun']:
                fun = res['fun']
                self.min_stats = res

        # 5. Get fit statistics
        # store the best fit parameter
        self.fit_stats['Param'] = self.min_stats['x']
        # store best fit data points evaluated over xdata
        self.fit_stats['Best fit'] = self.model.apply(self.abscissa,*self.min_stats['x'])
        # compute and store the covariance matrix of the fit using implementation of .fitting_helpers
        self.fit_stats['Cov'] = cov_fit_param(self.abscissa,self.ordinate_cov_inv,self.model,self.min_stats['x'],self.inv_acc)
        # compute and store the fit error
        self.fit_stats['Fit error'] = np.sqrt(np.diag(self.fit_stats['Cov']))
        # define the degrees of freedom
        dof = len(self.abscissa)-self.model.num_params
        # compute reduced chisq
        self.fit_stats['red chisq'] = self.chisq(self.fit_stats['Param']) / dof
        # compute p-value
        self.fit_stats['p-value']  = scipy.stats.chi2.sf(self.chisq(self.fit_stats['Param']),dof)
        if self.data is not None:
            # compute Akaike information criterion for normally distributed errors
            self.fit_stats['AIC'] = AIC_chisq(dof, self.fit_stats['red chisq'])
            # compute Akaike information criterion for small data sets
            self.fit_stats['AICc'] = AICc_chisq(dof, self.data.shape[0], self.fit_stats['red chisq'])

        # return the report this is later also accessible from the class
        return self.fit_stats

    def print_result(self,*args,**kwargs):
        out_str = "=======================================\n"
        out_str+= f"Reporting for model {self.model.__name__()}\n"
        out_str+= "=======================================\n"
        out_str+= "========= Best Fit Parameter: =========\n"
        for i_param in range(self.model.num_params):
            out_str+=f"{self.model.param_names[i_param]} = {self.fit_stats['Param'][i_param]:.6e} \u00B1 {self.fit_stats['Fit error'][i_param]: .6e}\n"

        out_str+= "========= Best Fit Covariance: ========\n"
        for i in range(self.model.num_params):
            for j in range(self.model.num_params):
                out_str+=f"{self.fit_stats['Cov'][i,j]: .2e}  "
            out_str+="\n"

        out_str+= "========= Best Fit \u1d61\u00B2: ================\n"
        out_str+= f"\u1d61\u00B2/dof = {self.fit_stats['red chisq']: .6e}\n"

        out_str+= "========= Best Fit p-value: ===========\n"
        out_str+= f"p-value = {self.fit_stats['p-value']: .6e}\n"
        if self.data is not None:
            out_str+= "========= Best Fit Akaike crit: =======\n"
            out_str+= f"AIC  = {self.fit_stats['AIC']: .6e}\n"
            out_str+= f"AICc = {self.fit_stats['AICc']: .6e}\n"

        out_str+= "========= Inverse Cov Accuracy: =======\n"
        out_str+= f"{self.fit_stats['Cov inv acc']:.6e}"

        print(out_str,*args,**kwargs)
