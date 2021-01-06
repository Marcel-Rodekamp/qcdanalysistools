import numpy as np
import qcdanalysistools.fitting as fitting
import qcdanalysistools.analysis as ana
import matplotlib.pyplot as plt

# number of data points i.e. gauge configurations
N = 212

# dimension i.e. size of temporal dimension
D = 48

# abscissa
x = np.array([x for x in range(D)])

# ordinate data
y = np.array([[ x_**2 for x_ in x ] for _ in range(N) ],dtype=np.float)
y += np.random.randn(N,D)


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

# ordinate variance with plain
ov_p = np.var(y,axis=0)
# ordinate variance with bootstrap
ov_b = ana.Bootstrap.var(y,bst_param,axis=0)
# ordinate variance with jackknife
ov_j = ana.Jackknife.var(y,jkn_param,axis=0)
# ordinate varince with blocking
ov_bl = ana.Blocking.var(y,blk_param,axis=0)

print(f"Ordinate data shape      = {y.shape}")
print(f"Abscissa shape           = {x.shape}\n")

print(f"Ordinate plain shape     = {o_p.shape}")
print(f"Ordinate bootstrap shape = {o_b.shape}")
print(f"Ordinate jackknife shape = {o_j.shape}")
print(f"Ordinate blocking shape  = {o_bl.shape}\n")

print(f"Ordinate variance plain shape     = {o_p.shape}")
print(f"Ordinate variance bootstrap shape = {o_b.shape}")
print(f"Ordinate variance jackknife shape = {o_j.shape}")
print(f"Ordinate variance blocking shape  = {o_bl.shape}\n")

# model
m = fitting.MonomialModel(t_A0=0,t_order=2)

print("Testing fitting base with data and plain average")
f = fitting.DiagonalLeastSquare(m,x,t_data=y)
if not np.allclose( f.ordinate,o_p ):
    raise RuntimeError(f"Ordinate computation failed \n {f.ordinate} \n {o_p}")
if not np.allclose( f.ordinate_var,ov_p ):
    raise RuntimeError(f"Ordinate variance computation failed \n {f.ordinate_var} \n {ov_p}")

# perform the fit
fit_res = f()

# just print fit_res nicely
f.print_result()

# plot the result for human eye check
plt.errorbar(x,o_p,yerr=np.sqrt(ov_p),fmt='*', label = "Avergaed Data")
plt.plot(x,fit_res['Best fit'],label="Best Fit")
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x,A) = A*x^2$")
plt.legend()
plt.show()
plt.clf()


# TODO:


print("Creating fitting base with data and bootstrap average", end="... ")
f = fitting.DiagonalLeastSquare(m,x,t_data=y,t_analysis_params=bst_param)
if not np.allclose( f.ordinate,o_b ):
    raise RuntimeError(f"Ordinate computation failed \n {f.ordinate} \n {o_b}")
if not np.allclose( f.ordinate_var,ov_b ):
    raise RuntimeError(f"Ordinate variance computation failed \n {f.ordinate_var} \n {ov_b}")
print("worked")

print("Creating fitting base with data and Jackknife average", end="... ")
f = fitting.DiagonalLeastSquare(m,x,t_data=y,t_analysis_params=jkn_param)
if not np.allclose( f.ordinate,o_j ):
    raise RuntimeError(f"Ordinate computation failed \n {f.ordinate} \n {o_j}")
if not np.allclose( f.ordinate_var,ov_j ):
    raise RuntimeError(f"Ordinate variance computation failed \n {f.ordinate_var} \n {ov_j}")
print("worked")

print("Creating fitting base with data and blocking average", end="... ")
f = fitting.DiagonalLeastSquare(m,x,t_data=y,t_analysis_params=blk_param)
if not np.allclose( f.ordinate,o_bl ):
    raise RuntimeError(f"Ordinate computation failed \n {f.ordinate} \n {o_bl}")
if not np.allclose( f.ordinate_var,ov_bl ):
    raise RuntimeError(f"Ordinate variance computation failed \n {f.ordinate_var} \n {ov_bl}")
print("worked")
