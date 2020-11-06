import numpy as np
r"""
    This file contains a method to compute the
        * akaike information criterion
"""

def AIC_chisq(t_dof, t_chisq):
    """
        t_dof: int
            Number of degrees of freedom i.e. number of parameters of the model
        t_chisq: float
            \u03C7\u00B2 of the model.

        Retruns: float
            AIC

        Assumptions:
            * normally distributed errors

        This function computes
        $$
            AIC = \chi^2 + 2k
        $$

        Following Sz. Borsanyi et. al.
            doi.org/10.1126/science.1257050
    """

    return t_chisq + 2*t_dof

def AICc_chisq(t_dof, t_datasize, *AIC_args, **AIC_kwargs):
    r"""
        t_dof: int
            Number of degrees of freedom i.e. number of parameters of the model
        t_datasize: int
            Size of the data set
        *AIC_args: arguments
            Arguments passt to t_AIC apart from t_dof
        **AIC_kwargs: Keyworded arguments
            Keyworded arguments passt to t_AIC apart

        Compute the Akaike information criterion for small data sets (n small)
        $$
            AICc = AIC_chi + \frac{2k(k+1)}{n-k-1}
        $$
        where and k is the #dof
    """
    return AIC_chisq(t_dof,*AIC_args,**AIC_kwargs) + (2*t_dof*(t_dof+1))/(t_datasize-t_dof-1)

def AIC_weights(t_AICs):
    r"""
        t_AICs: numpy.array (dim = 1)
            Set of Akaike criterions for different models
    """
    denom = np.sum( np.exp(-0.5*(t_AICs) ) )
    return np.exp( -0.5*(t_AICs) )/denom
