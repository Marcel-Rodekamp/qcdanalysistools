import numpy as np
from scipy.optimize import fsolve

def symmetrize(t_correlator):
    sym_correlator = np.zeros(t_correlator.shape[0]//2)
    Nt = t_correlator.shape[0]
    for t in range(t_correlator.shape[0]//2):
        sym_correlator[t] =  0.5*( t_correlator[t] + t_correlator[Nt-1-t] )

    return sym_correlator

def effective_mass_solver(t_t,t_sym_correlator,t_initial_guess):
    r"""!
        t: int
            Time slice point at which m_eff is computed
        t_sym_correlator: numpy.array
            Symmetrized correlator data
        t_initial_guess:
            Initial guess for the fsolve method

        This function solves the equation
        $$
            \frac{C_2(t)}{C_2(t+1)} - \frac{\cosh\left(m_eff * \left(t-\frac{N_t}{2}\right)\right)}{\cosh\left(m_eff \cdot \left(t+1-\frac{N_t}{2}\right)\right)} = 0
        $$
        for the effective mass at a given time slice point t.
    """
    upper_index = t_t+1 if t_t+1 < t_sym_correlator.size else 0

    # Equation to solve for effective_mass as lambda function
    effective_mass_func_cosh = lambda m_eff: \
        t_sym_correlator[t_t]/t_sym_correlator[upper_index]-np.cosh(m_eff*(t_t-t_sym_correlator.size))/np.cosh(m_eff*(t_t+1-t_sym_correlator.size))

    return fsolve(effective_mass_func_cosh,t_initial_guess)

def effective_mass(t_correlator,t_initial_guess, t_analysis_type, **analysis_kwargs):
    r"""
        t_correlator: numpy.ndarray
            Lattice QCD correlator data in the format
                t_correlator.shape = (configs,Nt)
            Assumptions:
                * The imainary part is neglegtable (will be omitted)
                * The correlator is periodic in temporal direction
        t_initial_guess: float
            For each gauge configuration the equation
                (C_2(t))/(C_2(t+1)) - cosh(m_eff * (t-Nt/2))/cosh(m_eff * (t+1-Nt/2)) = 0
            where C_2(t) is the 2 point correlator at a specific time slice t,
            Nt is the number of points in the temporal dimension and m_eff is the
            effective mass, has to be solved for m_eff.
            This value is the initial guess for the solver.
        t_analysis_type: string
            Determines the analysis type i.e. estimator and variance. Can be
                * Jackknife: qcdanalysistools.analysis.jackknife
                * Bootstrap: qcdanalysistools.analysis.bootstrap
                * Blocking:  qcdanalysistools.analysis.blocking
            Further specification and combinations are passed via
        **analysis_kwargs:
            keyworded arguments which are passed to the appropriate analysis tool
            determined by t_analysis_type

        Returns: numpy.array, numpy.array
            The effective_mass for half of all time slizes is returned aswell as
            well as cooresponding variances

        This function determines the effective mass per time slize (t) from a given
        correlator set. It is asuumed that the correlator set is real valued, or
        has neglegtable imaginary parts and is periodic in temporal direction.
        The effective mass in each time slice is determined by solving
        $$
            \frac{C_2(t)}{C_2(t+1)} - \frac{\cosh\left(m_eff * \left(t-\frac{N_t}{2}\right)\right)}{\cosh\left(m_eff \cdot \left(t+1-\frac{N_t}{2}\right)\right)} = 0
        $$
        And the analysis over configurations can be determined by either Jackknife,
        Bootstrap, and/or blocking following the standards of qcdanalysistools.

        Requirements:
            * numpy
            * qcdanalysistools (if Jackknife,Bootstrap and/or blocking)
            * scipy.optimize.fsolve
    """
    # project to the real part
    if np.iscomplex(t_correlator).any():
        t_correlator = t_correlator.real

    # effective masses for each config
    effective_mass = np.zeros(shape=(t_correlator.shape[0],t_correlator.shape[1]//2))

    # compute the effective mass on each configuration
    for conf_index in range(t_correlator.shape[0]):
        # symmetrize correlator only for this configuration
        sym_correlator = symmetrize(t_correlator[conf_index])

        # Solve for the effective mass in each time slice
        # Only use halfe of the time slices as the correlator is assumed to be periodic
        for t in range(t_correlator.shape[1]//2):

            # solve the abouve equation
            effective_mass[conf_index][t] = effective_mass_solver(t,sym_correlator,t_initial_guess)

    if t_analysis_type == "Plain":
        analysis = lambda data: ( np.average(data, axis=0),np.var(data,axis=0) )
    elif t_analysis_type == "Jackknife":
        from qcdanalysistools.analysis import jackknife
        analysis = lambda data: jackknife(data,**analysis_kwargs)
    elif t_analysis_type == "Bootstrap":
        from qcdanalysistools.analysis import bootstrap
        analysis = lambda data: bootstrap(data,**analysis_kwargs)
    elif t_analysis_type == "Blocking":
        from qcdanalysistools.analysis import blocking
        analysis = lambda data: blocking(data,**analysis_kwargs)
        pass
    else:
        print("## Warning: t_analysis_type =",t_analysis_type, "is not implemented, falling back to `Plain`")
        analysis = lambda data: ( np.average(data, axis=0),np.var(data,axis=0) )

    est, var = analysis(effective_mass)

    return est,var
