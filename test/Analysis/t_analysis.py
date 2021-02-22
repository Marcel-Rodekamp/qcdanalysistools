import qcdanalysistools.analysis as analysis
import numpy as np
import pathlib
import itertools

N = 100
D = 48

#mu,sigma = 0,0.1
lamda = 1/2

machinary = np.random.MT19937(np.random.SeedSequence(1234))
generator = np.random.Generator(machinary)

var = 2
mean = 0
mean_vec = np.ones(N)*mean
cov = np.zeros(shape=(N,N))

for i,j in itertools.product(range(N),repeat=2):
    if i==j:
        cov[i,i] = var
    if i>j:
        cov[i,j] = np.random.rand()/1000

for i,j in itertools.product(range(N),repeat=2):
    cov[i,j] = cov[j,i]

cov = cov@cov.transpose()
print(cov)

test_data = generator.multivariate_normal(mean=mean_vec,cov=cov)

bst_param = analysis.AnalysisParam(analysis.Bootstrap,
    data_size = N,
    N_bst     = 100,
    use_blocking = True,
    N_blk = 2,
    store_bst_samples = False,
    store_bst_samples_fn = pathlib.Path("results/analysis_test/bst_dat_store.h5")
)
param_jak = analysis.AnalysisParam(analysis.Jackknife,
    data_size = N,
    N_jak     = 10,
    use_blocking = True,
    N_blk     = 2
)

param_blk = analysis.AnalysisParam(analysis.Blocking,
    data_size = N,
    N_blk     = 10
)

param = param_blk

print(param)

print(f"True  mean = {mean :.2e}, True  Variance = {var:.2e}")
print(f"Numpy mean = {np.average(test_data):.2e}, Numpy Variance = {np.var(test_data):.2e}")
print(f"Tools mean = {analysis.estimator(param,test_data):.2e}, Tools Variance = {analysis.variance(param,test_data):.2e}")
#print(f"Tools Variance as Estimator = {analysis.estimator(param,test_data,t_observable=np.var):.2e}")
