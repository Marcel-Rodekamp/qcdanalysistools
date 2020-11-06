from .effective_mass import symmetrize

import numpy as np
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

r"""
    This file contains methods to extract 1st order energy levels from two point
    correlation functions C(t). Thus it is assumed that for $t\to\infty$
        $$
            C(t) = A_0 \cosh\left(\left( t - \frac{N_t}{2} \right) E_0\right)
        $$
    A periodic temporal direction is assumed
"""

def energy_firstorder(t_correlator, t_fit_range = None, t_A0 = 1, t_E0=1, t_analysis_type="Plain", t_plot_path=None, **analysis_kwargs):
    r"""
        t_correlator: numpy.ndarray
            Data array, containing the raw data of two point correlation function.
            It is assumed, that axis=0 represents the data points of the method
            and axis=1 represents the temportal extend $t\in[0,N_t-1]$.
        t_fit_range: tuple, default: None
            The fit is performed in the range [ t_fit_range[0], t_fit_range[1] ). (upper bound is excluded)
            If default (None) is given the fitrange is extended to the half axis
            $t\in[0,(N_t-1)//2]$.
        t_A0: float
            Initial guess of $A_0$ for the fit.
        t_E0: float
            Initial guess of $E_0$ for the fit.
        t_analysis_type: string, default: "Plain"
            Sates which analysis method is chosen to estimate the correlator data
            Possibilities are:
                * "Plain"     : numpy.average, numpy.variance
                * "Jackknife" : qcdanalysistools.analysis.jackknife
                * "Bootstrap" : qcdanalysistools.analysis.bootstrap
                * "Blocking"  : qcdanalysistools.analysis.blocking
            Further parameters are passed via **analysis_kwargs
        t_plot_path: string, default: None
            Path where the resulting plot should be stored. If None, no plot is
            generated.
        **analysis_kwargs: keyworded arguments
            Keyworded arguments passed to the desired analysis method

        Returns
            Proportionality $A_0$, First Level Energy $E_0$

        Requirements:
            * numpy
            * scipy.optimize
            * matplotlib.pyplot (optional)

        This function determines, from a raw set of 2 point correlator data, the
        first energy level $E_0$ and the proportionality $A_0$ of the correlator
        behaviour
        $$
            C(t) = A_0 \cosh\left(\left( t - \frac{N_t}{2} \right) E_0\right)
        $$
        for large times $t\in\$. This is achived by a fit. The resulting fit and
        data is plotted and eventually stored in `plot_path`
    """
    if t_analysis_type == "Plain":
        analysis = lambda data: ( np.average(data, axis=0),np.var(data,axis=0) )
    elif t_analysis_type == "Jackknife":
        from qcdanalysistools.analysis import jackknife
        analysis = lambda data: jackknife(data,**analysis_kwargs,axis=0)
    elif t_analysis_type == "Bootstrap":
        from qcdanalysistools.analysis import bootstrap
        analysis = lambda data: bootstrap(data,**analysis_kwargs,axis=0)
    elif t_analysis_type == "Blocking":
        from qcdanalysistools.analysis import blocking
        analysis = lambda data: blocking(data,**analysis_kwargs,axis=0)
        pass
    else:
        print("## Warning: t_analysis_type =",t_analysis_type, "is not implemented, falling back to `Plain`")
        analysis = lambda data: ( np.average(data, axis=0),np.var(data,axis=0) )

    t_correlator = symmetrize(t_correlator)

    # determine the correlator estimator
    corr_est,corr_var = analysis(t_correlator)
    # determine the error
    corr_err = np.sqrt(corr_var)
    # set time slize scale for readablility
    Nt = 2*corr_est.size
    # define temporal axis
    t_axis = np.array([t for t in range(corr_est.size)])
    # set fit range
    if t_fit_range is None:
        t_fit_range = t_axis
    else:
        t_fit_range = np.array( [t for t in range(t_fit_range[0],t_fit_range[1])] )
    # define fit model
    correlator_fct = lambda t,A_0,E_0 : \
        A_0*np.cosh((t -Nt/2)*E_0)
    # perform fit, Results: popt = [A_0,E_0], pcov = [[ Var(A_0,A_0),Var(A_0,E_0) ],
    #                                                 [ Var(E_0,A_0),Var(E_0,E_0) ]]
    popt,pcov = scipy.optimize.curve_fit(correlator_fct,
                                xdata = t_fit_range,
                                ydata = corr_est[t_fit_range[0]:t_fit_range[-1]+1],
                                p0    = [t_A0,t_E0],
                                sigma = corr_err[t_fit_range[0]:t_fit_range[-1]+1] )

    corr_fit = correlator_fct(t_fit_range, popt[0], popt[1])

    chisq, p_val = scipy.stats.chisquare(corr_fit,
                                        f_exp = corr_est[t_fit_range[0]:t_fit_range[-1]+1])

    report_string = "First level energy report:\n"
    report_string+= f"A_0 = {popt[0]} \u00B1 {np.sqrt(pcov[0][0])} \n"
    report_string+= f"E_0 = {popt[1]} \u00B1 {np.sqrt(pcov[1][1])} \n"
    report_string+= "Cov matrix = [ {0:.2e} , {1:.2e} ] \n".format(pcov[0][0],pcov[0][1])
    report_string+= "             [ {0:.2e} , {1:.2e} ] \n".format(pcov[0][0],pcov[0][1])
    report_string+= f"\u03C7\u00B2: {chisq} \n"
    report_string+= f"p-value: {p_val}"

    print(report_string)

    if t_plot_path is not None:
        plt.errorbar(t_axis,
                     corr_est,
                     fmt = '+',
                     yerr = corr_err,
                     label = r"$C(t)$")
        plt.plot(t_fit_range,
                 corr_fit,
                 label = r"$ C(t) = A_0 \cosh\left(\left( t - \frac{{N_t}}{{2}} \right) E_0\right)$, $_{{A_0 = {0:.2g},\, E_0 = {1:.2g} }}$".format(*popt))

        plt.grid()
        plt.xlabel("t")
        plt.ylabel(r"$C\left(t\right)$")
        plt.legend()
        plt.savefig(t_plot_path)

    return (*popt),pcov,corr_est,corr_var,corr_err
