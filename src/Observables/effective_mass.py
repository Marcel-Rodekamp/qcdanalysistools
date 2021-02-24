import numpy as np
from scipy.optimize import fsolve
from ..analysis import estimator,variance

def symmetrize(t_correlator):
    correlator = np.zeros(shape = (t_correlator.shape[0],t_correlator.shape[1]//2))
    Nt = t_correlator.shape[1]

    for i_ensamble, ensamble_ele in enumerate(t_correlator):
        correlator[i_ensamble][0] = ensamble_ele[0]
        correlator[i_ensamble][(Nt-1)//2] = ensamble_ele[(Nt-1)//2]
        for t in range(1, Nt//2-1):
            correlator[i_ensamble][t] =  0.5*( ensamble_ele[t] + ensamble_ele[-1-t] )

    return correlator

def effective_mass_cosh(t_correlator,t_initial_guess, t_analysis_params,Nt_half=24):
    r"""
        t_correlator: numpy.ndarray
            Lattice QCD correlator data in the format
                t_correlator.shape = (configs,Nt)
            Assumptions:
                * The imainary part is neglegtable (will be omitted)
        t_initial_guess: float
            For each gauge configuration the equation
                (C_2(t))/(C_2(t+1)) - cosh(m_eff * (t-Nt/2))/cosh(m_eff * (t+1-Nt/2)) = 0
            where C_2(t) is the 2 point correlator at a specific time slice t,
            Nt is the number of points in the temporal dimension and m_eff is the
            effective mass, has to be solved for m_eff.
            This value is the initial guess for the solver.
        t_analysis_params: AnalysisParams
            Determines the analysis type i.e. estimator and variance. Can be
                * Jackknife: qcdanalysistools.analysis.jackknife
                * Bootstrap: qcdanalysistools.analysis.bootstrap
                * Blocking:  qcdanalysistools.analysis.blocking

        Returns: numpy.array, numpy.array
            The effective_mass for half of all time slizes is returned aswell as
            well as cooresponding variances

        This function determines the effective mass per time slize (t) from a given
        correlator set. It is asuumed that the correlator set is real valued, or
        has neglegtable imaginary parts.
        The effective mass in each time slice is determined by solving
        $$
            \frac{C_2(t)}{C_2(t+1)} - \frac{\cosh\left(m_eff * \left(t-\frac{N_t}{2}\right)\right)}{\cosh\left(m_eff \cdot \left(t+1-\frac{N_t}{2}\right)\right)} = 0
        $$
        And the analysis over configurations can be determined by either Jackknife,
        Bootstrap, and/or blocking following the standards of qcdanalysistools.

        Requirements:
            * numpy
            * qcdanalysistools (if Jackknife,Bootstrap and/or Blocking)
            * scipy.optimize.fsolve
    """
    def effective_mass_solver(t_t,t_correlator,t_initial_guess,Nt_half=24):
        r"""!
            t: int
                Time slice point at which m_eff is computed
            t_correlator: numpy.array
                Correlator data
            t_initial_guess:
                Initial guess for the fsolve method

            This function solves the equation
            $$
                \frac{C_2(t)}{C_2(t+1)} - \frac{\cosh\left(m_eff * \left(t-\frac{N_t}{2}\right)\right)}{\cosh\left(m_eff \cdot \left(t+1-\frac{N_t}{2}\right)\right)} = 0
            $$
            for the effective mass at a given time slice point t.
        """
        upper_index = t_t+1 if t_t+1 < Nt_half else 0

        m_eff = np.zeros(shape=(t_correlator.shape[0]))

        # TODO: parallelize this loop.
        for i_ens in range(t_correlator.shape[0]):
            m_eff_fct = lambda m_eff: t_correlator[i_ens][t_t]/t_correlator[i_ens][upper_index]-np.cosh(m_eff*(t_t-Nt_half))/np.cosh(m_eff*(t_t+1-Nt_half))

            m_eff[i_ens] = fsolve(m_eff_fct,t_initial_guess)

        return m_eff

    def effective_mass_obs(t_correlator,t_initial_guess,Nt_half=24):
        r"""
            t_correlator: numpy.ndarray
                Correlator data shape = (#configs,Nt)
            This function is ment to act as an observable passt to the jackknife,bootstrap
            or blocking
        """

        effective_mass = np.zeros(shape=(t_correlator.shape[1]))

        for t in range(t_correlator.shape[1]):
            effective_mass[t] = np.average(effective_mass_solver(t,t_correlator,t_initial_guess,Nt_half))

        return effective_mass

    if t_analysis_params.analysis_type == "jackknife":
        from qcdanalysistools.analysis.Jackknife import jackknife as analysis
    elif t_analysis_params.analysis_type == "bootstrap":
        from qcdanalysistools.analysis.Bootstrap import bootstrap as analysis
    elif t_analysis_params.analysis_type == "blocking":
        from qcdanalysistools.analysis.Blocking import blocking as analysis
    else:
        raise ValueError(f"No such analysis type ({t_analysis_params.analysis_type})")

    if np.iscomplexobj(t_correlator):
        t_correlator = t_correlator.real

    est, var = analysis(t_data  = t_correlator,
                        t_params= t_analysis_params,
                        t_obs   = effective_mass_obs,
                        # **obs_kwargs
                        t_initial_guess = t_initial_guess,
                        Nt_half=Nt_half)

    return est,var

def effective_mass_log(t_correlator,t_analysis_params):
    r"""
        t_correlator: numpy.ndarray
            Lattice QCD correlator data in the format
                t_correlator.shape = (configs,Nt)
            Assumptions:
                * The imainary part is neglegtable (will be omitted)
        t_analysis_params: AnalysisParams
            Determines the analysis type i.e. estimator and variance. Can be
                * Jackknife: qcdanalysistools.analysis.jackknife
                * Bootstrap: qcdanalysistools.analysis.bootstrap
                * Blocking:  qcdanalysistools.analysis.blocking

        Returns: numpy.array, numpy.array
            The effective_mass for half of all time slizes is returned aswell as
            well as cooresponding variances

        This function determines the effective mass per time slize (t) from a given
        correlator set. It is asuumed that the correlator set is real valued, or
        has neglegtable imaginary parts.
        The effective mass in each time slice is determined by
        $$
            m_eff (t + 1/2) = \ln\left( \frac{C(t)}{C(t+1)} \right)
        $$
        And the analysis over configurations can be determined by either Jackknife,
        Bootstrap, and/or blocking following the standards of qcdanalysistools.

        Requirements:
            * numpy
            * qcdanalysistools (if Jackknife,Bootstrap and/or Blocking)
            * scipy.optimize.fsolve
    """

    def effective_mass_obs(t_correlator):

        m_eff = np.zeros(shape=(t_correlator.shape[1]-1,))

        for t in range(t_correlator.shape[1]-1):
            x_t = t_correlator[:,t]/t_correlator[:,t+1]
            # some exceptional configurations (nosiy ones) have x_t < 0
            # To asses these please comment the following two lines out.
            # This is accounted for using np.abs(x_t) below
            #if np.any(x_t<0):
            #    print(f"t={t}: {np.nonzero(x_t<0)}")

            m_eff[t] = np.average(np.log(np.abs(x_t))) #/(t+0.5)
        return m_eff

    if np.iscomplexobj(t_correlator):
        t_correlator = t_correlator.real

    m_eff_est = estimator(t_analysis_params,t_correlator,t_observable=effective_mass_obs)
    m_eff_var = variance (t_analysis_params,t_correlator,t_observable=effective_mass_obs)

    return m_eff_est,m_eff_var
