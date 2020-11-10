import numpy as np
import matplotlib.pyplot as plt
import qcdanalysistools as tools
import scipy

# for scipy
fit_model_scipy = lambda x,A: A*x**2
# for qcdanalysistools.fitting
fit_model = tools.fitting.MonomialModel(t_A0=1,t_order=2)

# number of data points i.e. gauge configurations
N_dat = 212
# x data length i.e. size of temporal dimension
Nx = 48

# x axis
x = np.array([x for x in range(Nx)])

# sample 2 dimensional y data randomly around a x**2 function
y = np.array([[ x_**2 for x_ in x ] for _ in range(N_dat) ],dtype=np.float)
y += np.random.randn(N_dat,Nx)

# sample 1 dimensional y data ToDo: Not yet implemented
#y = np.array( [ x_**2 for x_ in x ],dtype=np.float )
#y += np.random.randn(Nx)

# perform uncorrelated fit from qcdanalysistools.fitting
Fit = tools.fitting.LeastSquare(fit_model, t_ydata=y, t_xdata=x)
result = Fit.fit()

# perform uncorrelated fit from scipy as reference
popt,pcov = scipy.optimize.curve_fit(fit_model_scipy, xdata=x, ydata=np.average(y,axis=0),p0=(1,))

# print results and compare by hand
print("Minimization Stats: ======================================")
print(Fit.min_stats)
print("My fit Stats: ============================================")
for key in result:
    print(key +": ", result[key])
print("Scipy Stats: =============================================")
print("Param: ", popt)
print("Cov :", pcov)
print("Difference Stats: ========================================")
print("Difference Param :", popt - result['Param'])
print("Difference Cov :", pcov - result['Cov'])

# plot the fit to see if it actually worked
plt.plot(x,np.average(y,axis=0),'o',label="Raw Data")
plt.plot(x,fit_model_scipy(x,*popt),label="Scipy curve_fit")
plt.plot(x,result['Best fit'], label="LeastSquare")

plt.legend()
plt.grid()
plt.show()
