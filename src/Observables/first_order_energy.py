from .effective_mass import symmetrize
from ..fitting import FirstEnergyCoshModel

import matplotlib.pyplot as plt
import numpy as np

import itertools

def energy_firstorder( t_correlator, t_meff, t_A0, t_E0, t_fitter, t_fitter_params = dict(),  t_plateau_est_acc = 1e-4, t_plateau_area = 2):
    r"""
        t_correlator: np.ndarray
            Correlator data. It is assumed that it is unsymmetrized and might
            contain complex values assuming imag part is ~ 0 (will be discarded)
            The shape is assumed to be of order (#configs,Nt)
        t_meff: np.array
            Array containing the values of a m_eff solution from
                qcdanalysistools.observable.effective_mass
            It is used to estimate the plateau by the rule:
                t_0 (plateau start) at
                $$
                \frac{m_{eff}[t_0] - m_{eff}[t_0+1]}{m_{eff}[t_0+1] - m_{eff}[t_0+2]} - 1 < \varepsilon
                $$
                t_e (plateau end) at
                $$
                    \frac{m_{eff}[t_0+2] - m_{eff}[t_0+1]}{m_{eff}[t_0+1] - m_{eff}[t_0]} - 1 < \varepsilon
                $$
                where $\varepsilon = $ t_plateau_est_acc
        t_A0:
            A - Start parameter for the fit
        t_E0:
            E - Start parameter for the fit
        t_fitter: qcdanalysistools.fitting.FitBase (uninitialized)
            Pass the fitting method derived from FitBase. Note it must be
            uninitialized as it is be used to create the fitting class during the
            process.
        t_fitter_params: dict, default: dict()
            Parameters passed to the initialization of t_fitter at every instance
            of the fitting procedute.
            The initialization has the trace
            ```
            fitter = t_fitter(t_model    = fit_model,
                            t_abscissa = absicssa,
                            t_data     = t_correlator[:,t0_i:te_i],
                            **t_fitter_params
            )
            ```
        t_plateau_est_acc: float
            Deviation which the plateau might have. Used to estimate t_0,t_e from
            above as $\varepsilon$.
        t_plateau_area: int, default: 2
            Region where to improve the plateau using chisq/dof from the fits. i.e.
            We fit for $t \in \left[ t_0 - plateau\_area,\, t_0 + plateau\_area\right]$
            and $t_e$ similarly.

        This method computes the first order energy level from correlator data i.e.
            $$
            C(t,A,E) \approx A \cosh\left( \left(t-\frac{N_t}{2}\right) \cdot E \right)
            $$
        The algorithm proceeds as follows:

        1. Symmetrize correlator (projecting the imag part to 0, if given)
        2. Estimate start of plateau from m_eff
        3. Estimate end of plateau from m_eff
        4. Perform energy fit in ranges t0_i in [t0-t_plateau_area, t0+t_plateau_area]
                                       te_i in [te-t_plateau_area, te+t_plateau_area]
        5. The fit with smallest chisq/dof is accepted while the rest is discarded

        Tips to improve results:
        * Error estimation of correlated fits are more accurate though might be
          infeasible in calcultation if the covariance becomes to singular.
        * Try the sampled fits to get reduce bias of the estimator of the fit.
            * This may take significantly more computation time.

        Credits:
        This algorithm is inspired by Hasan, Green et.al.
        https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.114505
        """

    Nt = t_correlator.shape[1]

    # 1. preprocess the data
    try:
        t_correlator = t_correlator.real
    except:
        # assume t_correlator is already real
        pass

    t_correlator = symmetrize(t_correlator)

    # 2. Estimate start plateau
    # ToDo This is not a very robust way of finding a plateau, a more sophisticated way
    #      should come at some point!
    t0 = 0
    te = Nt//2
    for t in range(Nt//2-2):
        if np.abs((t_meff[t] - t_meff[t+1])/(t_meff[t+1] - t_meff[t+2]) - 1.) < t_plateau_est_acc:
            t0 = t
            break

    # 3. Estimate end plateau
    for t in range(Nt//2-3,0,-1):
        if np.abs((t_meff[t+2] - t_meff[t+1])/(t_meff[t+1] - t_meff[t]) - 1.) < t_plateau_est_acc:
            te = t
            break

    if t0 == 0:
        print("Warning: Couldn't estimate plateau start, fitting starts at t0 = 0!")
    else:
        print(f"Plateau start has been estimated to be t\u2080 = {t0}")

    if te == Nt//2:
        print(f"Warning: Couldn't estimate plateau end, fitting starts at t0 = {Nt//2}!")
    else:
        print(f"Plateau end has been estimated to be t\u2091 = {te}")

    # 4. Perform the fit in plateau ranges
    best_red_chisq = 1e+200 #np.Infinity
    final_result_fitter = None
    absicssa = np.arange(Nt//2)

    for t0_i,te_i in itertools.product(range(t0-t_plateau_area,t0+t_plateau_area+1),range(te-t_plateau_area,te+t_plateau_area+1)):
        print(f"Fitting with t\u2080 = {t0_i} to t\u2091 = {te_i}")
        if t0_i < 0:
            pass
        if te_i >= Nt:
            pass

        fit_model = FirstEnergyCoshModel(t_A0, t_E0, Nt)

        fitter = t_fitter(t_model    = fit_model,
                        t_abscissa = absicssa[t0_i:te_i],
                        t_data     = t_correlator[:,t0_i:te_i],
                        **t_fitter_params
        )

        # 5. find best fit result
        if fitter.fit()['red chisq'] < best_red_chisq:
            final_result_fitter = fitter
            fitter.fit_stats['Fitting range'] = (t0_i,te_i)


    final_result_fitter.print_result()
    print( "=========== Best Fit Range: ===========\n"
        + f"(t\u2080 = {fitter.fit_stats['Fitting range'][0]}, t\u2091 = {fitter.fit_stats['Fitting range'][1]})"
    )

    return final_result_fitter.fit_stats
