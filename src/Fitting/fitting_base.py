
r"""
    This file contains a base fitting class which is used in all subsequent fitting methods.
"""
import numpy as np
import itertools

class FitBase():
    r"""
        Base class to fit a given lattice data.
        The initialization preprocesses the data so that it can be used in the
        common minimization method of scipy.optimize.
    """
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
                    * Theta0:
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

            This class represents the base class to all fitting methods.
            It defines a ...

            Notes:

            * If t_ydata is one dimensional an extra dimension is opened representing
            a single gauge configuration. This merely assumes that the data is already
            averaged over all gauge configuration. It must be done as the covariance
            calculation assumes a 2-dimensional input.
            * cov_y and its inverse incorperate the correlation matrix i.e.
            $$
                \frac{CoV[ydata]_{i,j}}{\sqrt{\sigma_i\sigma_j}}
            $$
            This is prefered agains the covariance matrix as it is less singular
            * All classes deriving from this must implement
                self.fit()

        """
        # store the model: models are defined in model.py or selfdeveloped based
        # on qcdanalysistools.fitting.ModelBase
        self.model = t_model
        # Store the desired frequencies i.e. the data to fit against
        # if the data is not two dimensional e.g. (#axis,#gauge) raise ValueError
        if len(t_ydata.shape) != 2:
            raise ValueError(f"t_ydata must be 2-dimensional but is of shape {t_ydata.shape}")
        self.ydata = t_ydata

        # Store the argument of the model e.g. time axis.
        self.xdata = t_xdata

        # optional arguments which are passed to the scipy.optimize.minimization
        # method. For documentation please refer to
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        self.min_args = min_args
        self.min_kwargs = min_kwargs

        # store the report from the minimization procedure i.e. scipy.optimize.OptimizeResult
        self.min_stats = dict()

        # store the fit report from the fitting procedure i.e. dictionary filled
        # once the self.fit() method is called
        self.fit_stats = dict()

        # Data preprocessing ===================================================

        # freeze covariance matrix
        self.cov_y = np.cov(self.ydata.T)

        # compute its inverse
        self.cov_y_inv = np.linalg.inv(self.cov_y)

        # check that the inverse worked out
        res = np.linalg.norm(self.cov_y @ self.cov_y_inv - np.identity(self.cov_y.shape[0]))
        if res > pow(10,-8):
            raise ValueError(f"Matrix inverse of the ydata correlation matrix is not precise: residuum = {res}")

    def _cov(self):
        r"""
            This function implements the computation of the covariance of the parammeters
            i.e.
            $$
                Cov[\Theta] = \left( J_\Theta C^{-1} J_\Theta^T \right)^{-1}
            $$
            where J is the jacobian of the model in respect to the parameters $\Theta$
        """
        # get dimension of the array:
        # i.e. number of parameters
        num_params = self.model.num_params
        # e.g. number of time slices Nt
        num_axis_points = self.ydata.shape[1]

        # compute the parameter jacobian from the model i.e. df(x_i,Theta)/dTheta_a
        # The shape must be (num_params,num_axis_points)
        J = self.model.jac_param(self.xdata)

        if J.shape != (num_params,num_axis_points):
            raise ValueError(f"Jacobian of model {self.model} has wrong shape: {jac_model.shape}. Require a shape of ({(num_params,num_axis_points)})")

        # allocate memory for the inverse covariance matrix of dimension #params x #params
        cov_inv = np.zeros(shape=(num_params,num_params))

        for a,b in itertools.product(range(num_params),repeat=2):
            for t1,t2 in itertools.product(range(num_axis_points),repeat = 2):
                # compute matrix product J @ CoV[data] @ J^T
                cov_inv[a,b] += J[a,t1]*self.cov_y[t1,t2]*J[b,t2]

        # invert the expression
        cov = np.linalg.inv(cov_inv*self.ydata.shape[0])

        # check that the inverse worked out
        res = np.linalg.norm(cov @ cov_inv*self.ydata.shape[0] - np.identity(num_params))
        if res > pow(10,-8):
            raise ValueError(f"Matrix inverse of the fit parameter covariance matrix is not precise: residuum = {res}")

        return cov

    def fit(self,*args,**kwargs):
        raise NotImplementedError
