import numpy as np

import qcdanalysistools.analysis as tools

# data shape (#gauge,*dim_estimator)
data_shape = (212,8,4)

bst_params = tools.BootstrapParams(
    t_data_size = data_shape[0],
    t_num_subdatasets = 10,
    t_with_blocking = False,
    t_num_blocks = 10)

def simple_obs(x,axis):
    return np.power(np.average(x,axis=axis),2)

def complicated_obs(x,axis):
    if (x < 1).any():
        chisq = 0
    else:
        chisq = 1
    return np.power(np.average(x,axis=axis),2),chisq

def test_full_Bootstrap():
    test_data = np.ones( shape=data_shape )
    # Bootstrap if [1,1,1,] -> est = 1, var = 0

    est,var = tools.Bootstrap.bootstrap(test_data,bst_params,axis=0)

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
        print("Bootstrap method worked")
    else:
        print("Bootstrap method didn't work")

def test_simple_Bootstrap():
    test_data = np.ones( shape=data_shape )
    # Bootstrap if [1,1,1,] -> est = 1, var = 0

    est,var = tools.Bootstrap.bootstrap(test_data,bst_params,t_obs=simple_obs,axis=0)

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
        print("Bootstrap method worked")
    else:
        print("Bootstrap method didn't work")

def test_complicated_Bootstrap():
    test_data = np.ones( shape=data_shape )
    # Bootstrap if [1,1,1,] -> est = 1, var = 0

    obs_out = [None]*bst_params.num_blocks
    Theta_k = np.zeros(shape = (bst_params.num_blocks,*data_shape[1:]))

    for k in range(bst_params.num_blocks):
        obs_out[k] = complicated_obs(tools.Bootstrap.subdataset(test_data,k,bst_params),axis=0)
        Theta_k[k] = obs_out[k][0] # or however one accesses it

    est = tools.Bootstrap.skeleton_est(Theta_k,bst_params)
    var = tools.Bootstrap.skeleton_var(Theta_k,bst_params)

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
        print("Bootstrap method worked")
    else:
        print("Bootstrap method didn't work")

test_full_Bootstrap()
test_simple_Bootstrap()
test_complicated_Bootstrap()
