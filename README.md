# QCD Analysis Tools

This python library aims to offer analysis tools for QCD Data

## Currently Implemented

### Major Tools

* Computation of effective mass from 2-point correlation function
```    
def effective_mass(t_correlator,t_initial_guess, t_analysis_type, **analysis_kwargs)
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

## Requirements

* `numpy`
* `matplotlib.pyplot`
* lmfit

## Installation

```
python setup.py install (requires sudo)
```

## Licenses
This software package is accessible under the MIT license.
