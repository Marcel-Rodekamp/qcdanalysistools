import numpy as np
class ModelBase:
    r"""
        Base class of the fit models
        Each model inheriting from this must implement
            * self.apply(x,*Theta)
            * self.jac_param(x)
    """
    def __init__(self,t_num_params, t_Theta0):
        r"""
            t_num_params: int
                Number of parameters
            t_Theta0:
                Initial values for the fit

            Notes:
                For reference `self.Theta` is ment to keep current values
                so ensure that in each apply call `self.Theta` is set.

                Also try to set these at the end of the fit.
        """
        self.num_params = t_num_params
        self.Theta0 = t_Theta0
        self.Theta = t_Theta0

    def apply(self,x):
        raise NotImplementedError

    def jac(self,x):
        raise NotImplementedError

class MonomialModel(ModelBase):
    r"""
        Monomial model
        $$
            f_n(x,A) = A * x^n
        $$
    """
    def __init__(self,t_A0,t_order):
        r"""
            t_A0: float
                Initial guess of the parameter
            t_order: int
                Order of the monomial
        """
        super().__init__(t_num_params = 1, t_Theta0 = (t_A0,))
        self.order = t_order

    def apply(self,t_x,t_A):
        r"""
            t_x: numpy.array or float
                Argument of the model
            t_A: float
                Parameter which becomes fitted
        """
        # set current parameter
        self.Theta = (t_A,)
        return t_A * pow(t_x,self.order)

    def jac_param(self,t_x):
        """
            t_x: numpy.array or float
                Argument of the model

            Implements the jacobian of the model in respect to the parameters
        """
        return np.array( [ pow(t_x,self.order) ] )
