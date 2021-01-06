import numpy as np
import qcdanalysistools.fitting as fitting
import qcdanalysistools.analysis as ana

# number of data points i.e. gauge configurations
N = 212
# dimension i.e. size of temporal dimension
D = 48

# abscissa
x = np.array([x for x in range(D)])

# ordinate data
y = np.array( [[ *x ] for _ in range(N)] )

# bootstrap params
bst_param = ana.BootstrapParams(N,100)
# jackknife params
jkn_param = ana.JackknifeParams(N)
# blocking params
blk_param = ana.BlockingParams(N,50)

# ordinate with plain
o_p = np.average(y,axis=0)
# ordinate with bootstrap
o_b = ana.Bootstrap.est(y,bst_param,axis=0)
# ordinate with jackknife
o_j = ana.Jackknife.est(y,jkn_param,axis=0)
# ordinate with blocking
o_bl = ana.Blocking.est(y,blk_param,axis=0)

print(f"Ordinate data shape      = {y.shape}")
print(f"Ordinate plain shape     = {o_p.shape}")
print(f"Ordinate bootstrap shape = {o_b.shape}")
print(f"Ordinate jackknife shape = {o_j.shape}")
print(f"Ordinate blocking shape  = {o_bl.shape}")
print(f"Abscissa shape           = {x.shape}")

# model
m = fitting.MonomialModel(t_A0=0,t_order=1)

print("Creating fitting base with data and plain average", end="... ")
f = fitting.FitBase(m,x,t_data=y)
if not np.allclose( f.ordinate,o_p ):
    raise RuntimeError(f"Ordinate computation failed \n {f.ordinate} \n {o_p}")
print("worked")

print("Creating fitting base with data and bootstrap average", end="... ")
f = fitting.FitBase(m,x,t_data=y,t_analysis_params=bst_param)
if not np.allclose( f.ordinate,o_b ):
    raise RuntimeError(f"Ordinate computation failed \n {f.ordinate} \n {o_b}")
print("worked")

print("Creating fitting base with data and Jackknife average", end="... ")
f = fitting.FitBase(m,x,t_data=y,t_analysis_params=jkn_param)
if not np.allclose( f.ordinate,o_j ):
    raise RuntimeError(f"Ordinate computation failed \n {f.ordinate} \n {o_j}")
print("worked")

print("Creating fitting base with data and blocking average", end="... ")
f = fitting.FitBase(m,x,t_data=y,t_analysis_params=blk_param)
if not np.allclose( f.ordinate,o_bl ):
    raise RuntimeError(f"Ordinate computation failed \n {f.ordinate} \n {o_bl}")
print("worked")
