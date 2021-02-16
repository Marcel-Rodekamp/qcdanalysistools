r"""
    This file contains classes for least square fitting
        * Uncorrelated Fit (Diagonal approximation) : DiagonalLeastSquare
        * Correlated Fit                            : CorrelatedLeastSquare
        * Correlated Fit with eigenmode shift       : CorrelatedLeastSquare_EigenmodeShift
"""
import numpy as np
import scipy.optimize as opt
import scipy.stats
import itertools
from .fitting_base import FitBase
from .fitting_helpers import * # cov,cor,cov_fit_param,cov_fit_param_est
from qcdanalysistools.stats import AIC_chisq, AICc_chisq

class Sampled_DiagonalLeastSquare(FitBase):
    def __init__(self,t_model,t_abscissa,t_data,t_analysis_params):
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
        super().__init__(t_model,t_abscissa,t_data=t_data,t_analysis_params=t_analysis_params)
        # Note the ordinate is initialized with zeros as it needs to be recomputed for
        # every sample generated by the analysis type

        # backup the average of the ordinate data for the statistics
        self.ordinate_frozen = self.ordinate

        # compute the vairance of the ordinate data for the statistics
        # and perform some preparations for the fit algorithm
        if t_analysis_params.analysis_type == "bootstrap":
            from qcdanalysistools.analysis.Bootstrap import var,subdataset
            self.ordinate_var_frozen = var(self.data,self.analysis_params,axis=0)
            self.num_samples = self.analysis_params.num_subdatasets
            self.get_sample = lambda i_sample: subdataset(self.data,self.analysis_params)

        elif t_analysis_params.analysis_type == "jackknife":
            from qcdanalysistools.analysis.Jackknife import var,subdataset
            self.ordinate_var_frozen = var(self.data,self.analysis_params,axis=0)
            self.num_samples = self.analysis_params.num_subdatasets
            self.get_sample = lambda i_sample: subdataset(self.data,i_sample,self.analysis_params)

        elif t_analysis_params.analysis_type == "blocking":
            from qcdanalysistools.analysis.Blocking import var,subdataset
            self.ordinate_var_frozen = var(self.data,self.analysis_params,axis=0)
            self.num_samples = self.analysis_params.num_blocks
            self.get_sample = lambda i_sample: subdataset(self.data,i_sample,self.analysis_params)
        else:
            raise NotImplementedError(f"Sampled fitting is not implemented for analysis type {self.analysis_params.analysis_type}")

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
            1. Draw a sample from the original data
            2. Compute a new ordinate and ordinate variance from that sample
            3. For each minimization algorithm in scipy.optimize do
            4.      Minimize chisq.
            5.      if minimize succeeded add result to min_res_list
            6. Choose the best minimization from min_res_list
            7. repeat for all samples
            8. Average results+statistics
        """
        param_per_sample = np.zeros( shape = (self.num_samples,self.model.num_params) )
        for i_sample in range(self.num_samples):
            print(f"Fitting on Sample: {i_sample}/{self.num_samples}")
            # 1. Draw subdata set
            l_data = self.get_sample(i_sample)

            #2. compute ordinate and variance
            self.ordinate = np.average(l_data,axis=0)
            self.ordinate_var = np.var(l_data,axis=0)

            # 3 - 5 for each minimization method minimize chisq
            min_res_list = self._fit()

            # Warn if no minimization method succeeded
            if len(min_res_list) == 0:
                raise RuntimeWarning(f"No minimization technique worked for fitting sample {i_sample}. Try using different start parameters.")

            # 6. find the smallest chisq of all algorithms
            self.min_stats = min_res_list[0]
            fun = min_res_list[0]['fun']
            # TODO: Do we require a faster algorithm here? Try tree structure then.
            for res in min_res_list[1:]:
                if fun > res['fun']:
                    fun = res['fun']
                    self.min_stats = res

            # store the best fit parameters to average
            param_per_sample[i_sample,:] = self.min_stats['x']

        # 5. Get fit statistics
        # store the best fit parameter
        self.fit_stats['Param'] = np.average(param_per_sample,axis=0)
        # compute and store the fit error from the sampling
        self.fit_stats['Fit sampled error'] = np.sqrt(np.var(param_per_sample,axis=0))
        # artificially compute the covariance matrix of the parameter from the backed up data
        # estimate the covariance with <Theta_i Theta_j> - <Theta_i><Theta_j>
        self.fit_stats['Cov'] = cov_fit_param_est(param_per_sample,t_analysis_params=None)
        # compute and store the fit error
        self.fit_stats['Fit error'] = np.sqrt(np.diag(self.fit_stats['Cov']))
        # store best fit data points evaluated over xdata
        self.fit_stats['Best fit'] = self.model(self.abscissa,*self.fit_stats['Param'])
        # define the degrees of freedom
        dof = len(self.abscissa)-self.model.num_params
        # compute reduced chisq
        self.fit_stats['red chisq'] = self.chisq(self.fit_stats['Param']) / dof
        # compute p-value
        self.fit_stats['p-value']  = scipy.stats.chi2.sf(self.chisq(self.fit_stats['Param']),dof)
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
            out_str+=f"{self.model.param_names[i_param]} = {self.fit_stats['Param'][i_param]:.6e} \u00B1 {self.fit_stats['Fit error'][i_param]:.6e} ({self.fit_stats['Fit sampled error'][i_param]:.6e} : sample)\n"

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

class Sampled_CorrelatedLeastSquare(FitBase):
    def __init__(self,t_model,t_abscissa,t_data,t_analysis_params,t_frozen_cov_flag=False,t_inv_acc=1e-8):
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
            t_frozen_cov_flag: bool, default: False
                Freezing the covariance matrix (i.e. computing it once over the
                full data set and not again for each sample) can be turned on
                by setting this flag to True.
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
        super().__init__(t_model,t_abscissa,t_data=t_data,t_analysis_params=t_analysis_params)

        # we could allow to compute the covariance now and not touch it during fit
        # this would reduce computational cost and might be more stable due to
        # less inversion of nearly singular matrices by setting this value true
        self.frozen_cov_flag = t_frozen_cov_flag

        # store the accuracy of inversion
        self.inv_acc = t_inv_acc

        # freeze the ordinate ordinate data for the statistics
        self.ordinate_frozen = self.ordinate

        # compute the covairance of the ordinate data for the statistics
        self.ordinate_cov_frozen = cov(self.data,self.analysis_params)

        try:
            self.ordinate_cov_inv_frozen = np.linalg.inv(self.ordinate_cov_frozen)
        except:
            print(f"WARNING: Require SVD to invert covariance matrix.")
            u,w,v = np.linalg.svd(self.ordinate_cov_frozen)
            self.ordinate_cov_inv_frozen = np.dot(np.dot(np.transpose(v),np.diag(np.divide(np.ones(w.size),w,out=np.zeros(w.size),where=w<self.inv_acc**2))),np.transpose(u))

        # check that the inversion worked
        res_r = self.res(self.ordinate_cov_frozen @ self.ordinate_cov_inv_frozen)
        res_l = self.res(self.ordinate_cov_inv_frozen @ self.ordinate_cov_frozen)

        if res_r > self.inv_acc:
            raise RuntimeError(f"Failed to right invert the frozen covariance matrix: res = {res_r:.4e}")
        if res_l > self.inv_acc:
            raise RuntimeError(f"Failed to left invert the frozen covariance matrix: res = {res_l:.4e}")

        # if the frozen one is for the fit rename
        if self.frozen_cov_flag:
            self.ordinate_cov = self.ordinate_cov_frozen
            self.ordinate_cov_inv = self.ordinate_cov_inv_frozen

            if res_r > res_l:
                self.fit_stats['Cov inv acc'] = res_r
            else:
                self.fit_stats['Cov inv acc'] = res_l

        else:
            self.ordinate_cov = None
            self.ordinate_cov_inv = None

        # and perform some preparations for the fit algorithm
        if t_analysis_params.analysis_type == "bootstrap":
            from qcdanalysistools.analysis.Bootstrap import subdataset
            self.num_samples = self.analysis_params.num_subdatasets
            self.get_sample = lambda i_sample: subdataset(self.data,self.analysis_params)

        elif t_analysis_params.analysis_type == "jackknife":
            from qcdanalysistools.analysis.Jackknife import subdataset
            self.num_samples = self.analysis_params.num_subdatasets
            self.get_sample = lambda i_sample: subdataset(self.data,i_sample,self.analysis_params)

        elif t_analysis_params.analysis_type == "blocking":
            from qcdanalysistools.analysis.Blocking import subdataset
            self.num_samples = self.analysis_params.num_blocks
            self.get_sample = lambda i_sample: subdataset(self.data,i_sample,self.analysis_params)
        else:
            raise NotImplementedError(f"Sampled fitting is not implemented for analysis type {self.analysis_params.analysis_type}")

    def res(self,A):
        r"""
            Compute the residuum of the difference of A to the identity
        """
        return np.linalg.norm(A-np.identity(A.shape[0]))

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
            1. Draw a sample from the original data
            2. Compute a new ordinate and ordinate covariance from that sample
            3. For each minimization algorithm in scipy.optimize do
            4.      Minimize chisq.
            5.      if minimize succeeded add result to min_res_list
            6. Choose the best minimization from min_res_list
            7. repeat for all samples
            8. Average results+statistics
        """
        param_per_sample = np.zeros( shape = (self.num_samples,self.model.num_params) )
        for i_sample in range(self.num_samples):
            print(f"Fitting for sample: {i_sample}/{self.num_samples}")
            # 1. Draw subdata set
            l_data = self.get_sample(i_sample)

            #2. compute ordinate
            self.ordinate = np.average(l_data,axis=0)

            # we could allow to compute the covariance only once for the full
            # data set. This would reduce computational costs and increase stability
            # for some minor error in the fitting procedure.
            if not self.frozen_cov_flag:
                #2.1 compute covariance
                # use the function from qcdanalysistools.fitting.fitting_helpers
                self.ordinate_cov = cov(l_data)

                # invert the covariance
                try:
                    self.ordinate_cov_inv = np.linalg.inv(self.ordinate_cov)
                except:
                    print(f"WARNING: Require SVD to invert covariance matrix.")
                    u,w,v = np.linalg.svd(self.ordinate_cov)
                    self.ordinate_cov_inv = np.dot(np.dot(np.transpose(v),np.diag(np.divide(np.ones(w.size),w,out=np.zeros(w.size),where=w<self.inv_acc**2))),np.transpose(u))

                # check that the inversion worked
                res_r = self.res(self.ordinate_cov @ self.ordinate_cov_inv)
                res_l = self.res(self.ordinate_cov_inv @ self.ordinate_cov)
                if res_r > self.inv_acc:
                    raise RuntimeError(f"Failed to right invert the covariance matrix: res = {res_r:.4e}")
                if res_l > self.inv_acc:
                    raise RuntimeError(f"Failed to left invert the covariance matrix: res = {res_l:.4e}")

            # 3 - 5 for each minimization method minimize chisq
            min_res_list = self._fit()

            # Warn if no minimization method succeeded
            if len(min_res_list) == 0:
                raise RuntimeWarning(f"No minimization technique worked for fitting sample {i_sample}. Try using different start parameters.")
                continue

            # 6. find the smallest chisq of all algorithms
            self.min_stats = min_res_list[0]
            fun = min_res_list[0]['fun']
            # TODO: Do we require a faster algorithm here? Try tree structure then.
            for res in min_res_list[1:]:
                if fun > res['fun']:
                    fun = res['fun']
                    self.min_stats = res

            # store the best fit parameters to average
            param_per_sample[i_sample,:] = self.min_stats['x']

        # 5. Get fit statistics
        # store the best fit parameter
        self.fit_stats['Param'] = np.average(param_per_sample,axis=0)
        # compute and store the fit error from the sampling
        self.fit_stats['Fit sampled error'] = np.sqrt(np.var(param_per_sample,axis=0))
        # artificially compute the covariance matrix of the parameter from the backed up data
        self.fit_stats['Cov'] = cov_fit_param_est(param_per_sample,t_analysis_params=None)
        # compute and store the fit error
        self.fit_stats['Fit error'] = np.sqrt(np.diag(self.fit_stats['Cov']))
        # store best fit data points evaluated over xdata
        self.fit_stats['Best fit'] = self.model(self.abscissa,*self.fit_stats['Param'])
        # define the degrees of freedom
        dof = len(self.abscissa)-self.model.num_params
        # compute reduced chisq
        self.fit_stats['red chisq'] = self.chisq(self.fit_stats['Param']) / dof
        # compute p-value
        self.fit_stats['p-value']  = scipy.stats.chi2.sf(self.chisq(self.fit_stats['Param']),dof)
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
            out_str+=f"{self.model.param_names[i_param]} = {self.fit_stats['Param'][i_param]:.6e} \u00B1 {self.fit_stats['Fit error'][i_param]:.6e} ({self.fit_stats['Fit sampled error'][i_param]:.6e} : sample)\n"

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

        if self.frozen_cov_flag:
            out_str+= "========= Inverse Cov Accuracy: =======\n"
            out_str+= f"{self.fit_stats['Cov inv acc']:.6e}"

        print(out_str,*args,**kwargs)
