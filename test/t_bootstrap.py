import numpy as np

import qcdanalysistools.analysis as ana

do_print = True

data_shape = (102,12)

def test_estimator():
    def obs(x,alpha=2):
        return np.power(np.average(x,axis=0),alpha)

    test_data = np.ones( shape=data_shape )
    vali_data = 1

    res = ana.bootstrap_est(
        test_data,
        t_num_subdata_sets = 50,
        t_obs = obs,
        alpha=20
    )

    if do_print:
        print("Test data:",test_data)
        print("Result   :",res)
        print("Valid res:",vali_data - res)

    if np.all(vali_data == res):
        print("Estimator Works")

def test_variance():

    def obs(x,alpha=2):
        return np.power(np.average(x,axis=0),alpha)

    test_data = np.ones( shape=data_shape )
    vali_data = 0

    res = ana.bootstrap_var(
        test_data,
        t_num_subdata_sets = 50,
        t_obs = obs,
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

    est,var = ana.jackknife(
        test_data,
        t_obs = obs,
        t_n = 1,
        t_random_leaveout = False,
        t_num_ran_indices = 20,
        t_blocked = False,
        t_num_blocks = True,
        alpha=20
    )

    if do_print:
        print("Test data:",test_data)
        print("Estimator:",est)
        print("Variance :",var)
        print("ValidateE:",vali_est-est)
        print("ValidateV:",vali_var-var)

    if np.all(vali_est == est) and np.all(vali_var == var):
        print("Bootstrap Works")

test_estimator()
print("")
test_variance()
print("")
test_jackknife()
