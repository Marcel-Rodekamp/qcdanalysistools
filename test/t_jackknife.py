import numpy as np

import qcdanalysistools.analysis as tools


# data shape (#gauge,*dim_estimator)
data_shape = (212,8,4)

jkk_params = tools.JackknifeParams(
    t_data_size = data_shape[0],
    t_n = 1,
    t_random_leaveout = False,
    t_num_ran_indices = 106,
    t_with_blocking   = False,
    t_num_blocks      = 10)

def simple_obs(x,axis):
    return np.power(np.average(x,axis=axis),2)

def complicated_obs(x,axis):
    if (x < 1).any():
        chisq = 0
    else:
        chisq = 1
    return np.power(np.average(x,axis=axis),2),chisq

def test_full_jackknife():
    test_data = np.ones( shape=data_shape )
    # jackknife if [1,1,1,] -> est = 1, var = 0

    est,var = tools.Jackknife.jackknife(test_data,jkk_params,axis=0)

    print("Estimator shape:",est.shape)
    print("Variance shape :",var.shape)

    worked_flag = True
    if (est != 1).all():
        worked_flag = False
        print(f"Estimator should be (1) but is {est}.")
    if (var != 0).all():
        worked_flag = False
        print(f"Variance should be (0) but is {var}.")

    if worked_flag:
        print("Jackknife method worked")
    else:
        print("Jackknife method didn't work")

def test_simple_jackknife():
    test_data = np.ones( shape=data_shape )
    # jackknife if [1,1,1,] -> est = 1, var = 0

    est,var = tools.Jackknife.jackknife(test_data,jkk_params,t_obs=simple_obs,axis=0)

    print("Estimator shape:",est.shape)
    print("Variance shape :",var.shape)

    worked_flag = True
    if (est != 1).all():
        worked_flag = False
        print(f"Estimator should be (1) but is {est}.")
    if (var != 0).all():
        worked_flag = False
        print(f"Variance should be (0) but is {var}.")

    if worked_flag:
        print("Jackknife method worked")
    else:
        print("Jackknife method didn't work")

def test_complicated_jackknife():
    test_data = np.ones( shape=data_shape )
    # jackknife if [1,1,1,] -> est = 1, var = 0

    full_obs_out = complicated_obs(test_data,axis=0)
    Theta_hat = full_obs_out[0]

    obs_out = [None]*jkk_params.num_subdatasets
    Theta_k = np.zeros(shape = (jkk_params.num_subdatasets,*data_shape[1:]))

    for k in range(jkk_params.num_subdatasets):
        obs_out[k] = complicated_obs(tools.Jackknife.subdataset(test_data,k,jkk_params),axis=0)
        Theta_k[k] = obs_out[k][0] # or however one accesses it

    est = tools.Jackknife.skeleton_est(Theta_k,Theta_hat,jkk_params)
    var = tools.Jackknife.skeleton_var(Theta_k,Theta_hat,jkk_params)

    print("Estimator shape:",est.shape)
    print("Variance shape :",var.shape)

    worked_flag = True
    if (est != 1).all():
        worked_flag = False
        print(f"Estimator should be (1) but is {est}.")
    if (var != 0).all():
        worked_flag = False
        print(f"Variance should be (0) but is {var}.")

    if worked_flag:
        print("Jackknife method worked")
    else:
        print("Jackknife method didn't work")

test_full_jackknife()
test_simple_jackknife()
test_complicated_jackknife()
