import numpy as np

import qcdanalysistools.analysis as tools

# data shape (#gauge,*dim_estimator)
data_shape = (212,8)

blk_params = tools.BlockingParams(
    t_data_size = data_shape[0],
    t_num_blocks = 10)

def simple_obs(x,axis):
    return np.power(np.average(x,axis=axis),2)

def complicated_obs(x,axis):
    if (x < 1).any():
        chisq = 0
    else:
        chisq = 1
    return np.power(np.average(x,axis=axis),2),chisq

def test_full_Blocking():
    test_data = np.ones( shape=data_shape )
    # Blocking if [1,1,1,] -> est = 1, var = 0

    est,var = tools.Blocking.blocking(test_data,blk_params,axis=0)

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
        print("Blocking method worked")
    else:
        print("Blocking method didn't work")

def test_simple_Blocking():
    test_data = np.ones( shape=data_shape )
    # Blocking if [1,1,1,] -> est = 1, var = 0

    est,var = tools.Blocking.blocking(test_data,blk_params,t_obs=simple_obs,axis=0)

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
        print("Blocking method worked")
    else:
        print("Blocking method didn't work")

def test_complicated_Blocking():
    test_data = np.ones( shape=data_shape )
    # Blocking if [1,1,1,] -> est = 1, var = 0

    full_obs_out = complicated_obs(test_data,axis=0)

    obs_out = [None]*blk_params.num_blocks
    Theta_k = np.zeros(shape = (blk_params.num_blocks,*data_shape[1:]))

    for k in range(blk_params.num_blocks):
        obs_out[k] = complicated_obs(tools.Blocking.subdataset(test_data,k,blk_params),axis=0)
        Theta_k[k] = obs_out[k][0] # or however one accesses it

    est = tools.Blocking.skeleton_est(Theta_k,blk_params)
    var = tools.Blocking.skeleton_var(Theta_k,blk_params)

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
        print("Blocking method worked")
    else:
        print("Blocking method didn't work")


test_full_Blocking()
test_simple_Blocking()
test_complicated_Blocking()
