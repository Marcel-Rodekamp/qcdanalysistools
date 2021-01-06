import numpy as np
class ModelBase:
    def __init__(self,t_num_params, t_params0, t_param_names):
        r"""
            t_num_params: int
                Number of parameters of the model
            t_params0: tuple
                Initial values of parameters. Must be of shape (t_num_params,)
            t_parameter_names: tuple of string
                Strings representing the parameters

            Base class for any model passed to qcdanalysistools.fitting

            Notes:
                * Each model inheriting from this must implement
                    * self.apply(self,x,*args,**kwargs)
                        * the model function
                        --> evaluated in self.__call__
                    * self.grad_param(self,x,*args,**kwargs)
                        * jacobian array in respect to the parameters
                    * self.hess_param(self,x,*args,**kwargs)
                        * hessian matrix in respect to the parameters
                    * self.__name__(self)
                        * String describing the model
        """
        self.num_params = t_num_params

        if len(t_params0) != t_num_params:
            raise ValueError(f"Count of initial values t_params0 ({t_params0}) does not match the number of parameters ({t_num_params})")

        # It is convenient to store the initial parameters. If the fitting is restarted
        self.params0 = t_params0

        self.param_names = t_param_names

    def apply(self,x,*args,**kwargs):
        raise NotImplementedError

    def grad_param(self,x,*args,**kwargs):
        raise NotImplementedError

    def hess_param(self,x,*args,**kwargs):
        raise NotImplementedError

    def __call__(self,x,*args,**kwargs):
        return self.apply(x,*args,**kwargs)

class MonomialModel(ModelBase):
    def __init__(self,t_A0,t_order):
        r"""
            t_A0: float
                Initial parameter
            t_order: int
                Order of the monomial

            Monomial Model:
                $$
                    f(x,A) = A * x^n
                $$
                where $n$ is the order
        """
        super().__init__(t_num_params = 1, t_params0 = (t_A0,), t_param_names = ("A",))

        self.order = t_order

    def __name__(self):
        return f"f(x,A) = A x^{self.order}"

    def apply(self,t_x,t_A):
        r"""
            t_x: numpy.array
                Argument of the model function (abscissa)
            t_A: float
                Parameter which becomes fitted
        """
        return t_A * pow(t_x,self.order)

    def grad_param(self,t_x,t_A):
        """
            t_x: numpy.array
                Argument of the model function (abscissa)
            t_A: float
                Parameter which becomes fitted
        """
        return np.array( [ pow(t_x,self.order) ] )

    def hess_param(self,t_x,t_A):
        """
            t_x: numpy.array
                Argument of the model function (abscissa)
            t_A: float
                Parameter which becomes fitted
        """
        # note the result must always be a num_params x num_params matrix
        # thus the inconvenient return code here.
        return np.array([[np.zeros(shape=t_x.shape)]])

class FirstEnergyCoshModel(ModelBase):
    def __init__(self,t_A0,t_E0,t_Nt):
        r"""
            t_A0: float
                Initial guess of the scaling parameter
            t_E0: float
                Initial guess of energy parameter
            t_Nt: int
                Temporal extend

            First energy cosh model is given by
                $$
                f(t,A,E) = A * cosh( (t-Nt/2)*E )
                $$
            This is used to extract first energy level (E) from 2-point-correlators
            For details see e.g.
                Christof Gattringer and Christian Lang.
                Quantum chromodynamics on the lattice: an introductory presentation.
                Vol. 788. Springer Science & Business Media, 2009.
        """
        super().__init__(t_num_params = 2, t_params0 = (t_A0,t_E0,), t_param_names = ("A","E"))
        self.Nt_half = t_Nt/2

    def apply(self,t_x,t_A,t_E):
        r"""
            t_x: numpy.array
                Argument of the model function (abscissa)
            t_A: float
                Scaling parameter
            t_E: float
                First energy level
        """
        return t_A * np.cosh( (t_x - self.Nt_half) * t_E )

    def grad_param(self,t_x,t_A,t_E):
        r"""
            t_x: numpy.array
                Argument of the model function (abscissa)
            t_A: float
                Scaling parameter
            t_E: float
                First energy level
        """
        buf= (t_x-self.Nt_half)

        dA = np.cosh(buf*t_E)
        dE = t_A*buf*np.sinh(buf*t_E)

        return np.array( [ dA,dE ] )

    def hess_param(self,t_x,t_A,t_E):
        r"""
            t_x: numpy.array
                Argument of the model function (abscissa)
            t_A: float
                Scaling parameter
            t_E: float
                First energy level
        """
        buf = (t_x-self.Nt_half)

        dA2 = np.zeros(t_x.size)
        dE2 = t_A * np.square(buf) * np.cosh( buf * t_E )
        dAdE = buf * np.sinh( buf * t_E ) # = dEdA

        return np.array( [ [dA2,dAdE],[dAdE,dE2] ] )

    def __name__(self):
        return f"f(t,A,E) = A*cosh( (t-Nt/2)*E )"
