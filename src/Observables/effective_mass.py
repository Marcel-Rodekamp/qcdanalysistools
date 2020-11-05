import numpy as np
from scipy.optimize import fsolve

def symmetrize(t_correlator):
    sym_correlator = np.zeros(shape = (t_correlator.shape[0],t_correlator.shape[1]//2))
    Nt = t_correlator.shape[1]

    for i_ensamble, ensamble_ele in enumerate(t_correlator):
        for t in range(Nt//2):
            sym_correlator[i_ensamble][t] =  0.5*( ensamble_ele[t] + ensamble_ele[Nt-1-t] )

    return sym_correlator

def effective_mass_solver(t_t,t_sym_correlator,t_initial_guess,Nt_half=24):
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
    upper_index = t_t+1 if t_t+1 < Nt_half else 0

    m_eff = np.zeros(shape=(t_sym_correlator.shape[0]))

    # TODO: parallelize this loop.
    for i_ens in range(t_sym_correlator.shape[0]):
        m_eff_fct = lambda m_eff: t_sym_correlator[i_ens][t_t]/t_sym_correlator[i_ens][upper_index]-np.cosh(m_eff*(t_t-Nt_half))/np.cosh(m_eff*(t_t+1-Nt_half))

        m_eff[i_ens] = fsolve(m_eff_fct,t_initial_guess)

    return m_eff

def effective_mass_obs(t_symmetrized_correlator,t_initial_guess):
    r"""
        t_sym_correlator: numpy.ndarray
            Correlator data symmetrized in the shape = (#configs,Nt)
        This function is ment to act as an observable passt to the jackknife,bootstrap
        or blocking
    """

    effective_mass = np.zeros(shape=(t_symmetrized_correlator.shape[1]))

    for t in range(t_symmetrized_correlator.shape[1]):
        effective_mass[t] = np.average(effective_mass_solver(t,t_symmetrized_correlator,t_initial_guess))

    return effective_mass

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

    if t_analysis_type == "Plain":
        analysis = lambda data: ( np.average(data, axis=0),np.var(data,axis=0) )
    elif t_analysis_type == "Jackknife":
        from qcdanalysistools.analysis import jackknife
        analysis = lambda data: jackknife(data,**analysis_kwargs,t_obs=effective_mass_obs,t_initial_guess=t_initial_guess)
    elif t_analysis_type == "Bootstrap":
        from qcdanalysistools.analysis import bootstrap
        analysis = lambda data: bootstrap(data,**analysis_kwargs,t_obs=effective_mass_obs,t_initial_guess=t_initial_guess)
    elif t_analysis_type == "Blocking":
        from qcdanalysistools.analysis import blocking
        analysis = lambda data: blocking(data,**analysis_kwargs,t_obs=effective_mass_obs,t_initial_guess=t_initial_guess)
        pass
    else:
        print("## Warning: t_analysis_type =",t_analysis_type, "is not implemented, falling back to `Plain`")
        analysis = lambda data: ( np.average(data, axis=0),np.var(data,axis=0) )

    # project to the real part
    print("Projected to Real")
    t_correlator = t_correlator.real

    # symmetrize the correlator of each element in the ensamble
    sym_correlator = symmetrize(t_correlator)

    est, var = analysis(sym_correlator)

    return est,var
