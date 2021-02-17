# QCD Analysis Tools

This python library aims to offer analysis tools for QCD Data

## Currently Implemented

### Major Tools

* Computation of effective mass from 2-point correlation function
```    
def effective_mass(t_correlator,t_initial_guess, t_analysis_type, **analysis_kwargs)
```
* First order energy by `C(t) ~ A cosh((t-Nt/2)*E )`
```
def energy_firstorder( t_correlator, t_meff, t_A0, t_E0, t_fitter, t_fitter_params = dict(),  t_plateau_est_acc = 1e-4, t_plateau_area = 2)
```

### Minor Tools

* Jackknife (with and without blocking)
```
def jackknife(t_data, t_n = 1, t_random_leaveout = False, t_num_ran_indices=None, t_blocked = False, t_num_blocks = None)
```
* Bootstrap (with and without blocking)
```
def bootstrap(t_data,t_num_leave_outs, t_blocked = False, t_num_blocks = None)
```
* Blocking
```
def blocking(t_data, t_num_blocks = 2) # estimator and variances
def var_per_num_blocks(t_data,t_num_blocks_range = None) # block number analysis
```
* Fitting
    * Least Square Diagonal Approximation
    ```
        class DiagonalLeastSquare(t_model,t_abscissa,t_data=None,t_ordinate=None,t_ordinate_var=None,t_analysis_params=None)
    ```
        * fit: `Fit_Results = DiagonalLeastSquare.fit()`
        * print: `DiagonalLeastSquare.print_result(*args,**kwargs)`
    * Least Square Correltated
    ```
        class CorrelatedLeastSquare(t_model,t_abscissa,t_data=None,t_ordinate=None,t_ordinate_cov=None,t_analysis_params=None, t_inv_acc=1e-8)
    ```
        * fit: `Fit_Results = DiagonalLeastSquare.fit()`
        * print: `DiagonalLeastSquare.print_result(*args,**kwargs)`
    * Sampled Least Square Diagonal Approximation
    ```
        class Sampled_DiagonalLeastSquare(t_model,t_abscissa,t_data,t_analysis_params)
    ```
        * fit: `Fit_Results = DiagonalLeastSquare.fit()`
        * print: `DiagonalLeastSquare.print_result(*args,**kwargs)`
    * Sampled Least Square Correltated
    ```
        class Sampled_CorrelatedLeastSquare(t_model,t_abscissa,t_data,t_analysis_params,t_frozen_cov_flag=False,t_inv_acc=1e-8)
    ```
        * set `t_frozen_cov_flag = True` to compute the convariance matrix once over full data and do not recompute it on every sample. Might increase
        stability and greatly increases speed.
        * fit: `Fit_Results = DiagonalLeastSquare.fit()`
        * print: `DiagonalLeastSquare.print_result(*args,**kwargs)`
*  Akaike information criterion
```
def AIC_chisq(t_dof, t_chisq)
def AICc_chisq(t_dof, t_datasize, *AIC_args, **AIC_kwargs)
def AIC_weights(t_AICs)
```
* dimension transformations
    * mass: `def  massMeV(t_mlat, t_a, t_mlat_err = None, t_a_err = None)`
        * Note: Either both or no err parameter must be given.


## Requirements

* `scipy`
* `itertools`
* `warnings`
* `pathlb`
* `numpy`
* `matplotlib.pyplot`
* `aff` USQCDs python aff library (Optional)
    * only required if `qcdanalysistools.fileIO.aff` is used.
    * needs to be installed beforehand

## Installation

```
python setup.py install --user
or
sudo python setup.py install
```

## Licenses
This software package is accessible under the MIT license.
