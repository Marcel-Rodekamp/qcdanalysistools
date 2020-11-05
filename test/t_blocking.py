import numpy as np

import qcdanalysistools.analysis as ana

do_print = True

data_shape = (102,12)
num_blocks = 10

def test_estimator():
    # define trivial observable
    def obs(x,alpha=2):
        return np.power(np.average(x,axis=0),alpha)

    test_data = np.ones( shape=data_shape )
    vali_data = 1 #np.ones(  )

    res = ana.blocking_est(
        test_data,
        t_obs = obs,
        t_num_blocks = num_blocks,
        alpha=20
    )

    if do_print:
        print("Test data:",test_data)
        print("Result   :",res)
        print("Valid res:",vali_data - res)

    if np.all(vali_data == res):
        print("Estimator Works")

def test_variance():
    # define trivial observable
    def obs(x,alpha=2):
        return np.power(np.average(x,axis=0),alpha)

    test_data = np.ones( shape=data_shape )
    vali_data = 0

    res = ana.blocking_var(
        test_data,
        t_obs = obs,
        t_num_blocks = num_blocks,
        alpha=20
    )

    if do_print:
        print("Test data:",test_data)
        print("Result   :",res)
        print("Valid res:",vali_data - res)

    if np.all(vali_data == res):
        print("Variance Works")

def test_jackknife():
    # define trivial observable
    def obs(x,alpha=2):
        return np.power(np.average(x,axis=0),alpha)

    test_data = np.ones( shape=data_shape )
    vali_var = 0
    vali_est = 1

    est,var = ana.blocking(
        test_data,
        t_obs = obs,
        t_num_blocks = num_blocks,
        alpha=20
    )

    if do_print:
        print("Test data:",test_data)
        print("Estimator:",est)
        print("Variance :",var)
        print("ValidateE:",vali_est-est)
        print("ValidateV:",vali_var-var)

    if np.all(vali_est == est) and np.all(vali_var == var):
        print("Blocking Works")

test_estimator()
print("")
test_variance()
print("")
test_jackknife()
