import numpy as np
import qcdanalysistools as tools

Nt = 6
data = np.random.randn(1000,Nt)

# for different analysis style set Jackknife,Bootstrap or Blocking
params = tools.analysis.BootstrapParams(
    t_data_size = data.shape[0],
    t_num_subdatasets = 1000,
    t_with_blocking = True,
    t_num_blocks = 50)

#params = tools.analysis.JackknifeParams(
#    t_data_size = data.shape[0],
#    t_n = 1,
#    t_random_leaveout = False,
#    t_num_ran_indices = 106,
#    t_with_blocking   = True,
#    t_num_blocks      = 50)

#params = tools.analysis.BlockingParams(
#    t_data_size = data.shape[0],
#    t_num_blocks = 20)

#params = None

cov = tools.fitting.cov(data,params)


print("cov.shape =",cov.shape)
print("cov       =",cov)
