import numpy as np

# https://pdg.lbl.gov/2010/reviews/rpp2010-rev-phys-constants.pdf , last visited 15.02.2021 15:00 CET
hbarXc = 197.3269631 # MeV fm
hbarXc_err = 4.9e-8 # MeV fm

def massMeV(t_mlat, t_a, t_mlat_err = None, t_a_err = None):
    m = t_mlat/t_a * hbarXc

    if t_mlat_err is None and t_a_err is None:
        return m, None
    else:
        # gaussian error propagation i.e. variance formular
        m_err = np.sqrt( (hbarXc**2 / t_a**2)             * t_mlat_err**2  \
                       + (hbarXc**2 * t_mlat**2 / t_a**4) * t_a_err**2     \
                       + (t_mlat/t_a)                     * hbarXc_err**2)

        return m,m_err
