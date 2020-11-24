import numpy as np
import matplotlib.pyplot as plt
import qcdanalysistools as tools
import scipy

# for scipy
fit_model_scipy = lambda x,A: A*x**2
# for qcdanalysistools.fitting
fit_model = tools.fitting.FirstEnergyCoshModel(t_A0=1.9,t_E0=0.55,t_Nt=48)

# number of data points i.e. gauge configurations
N_dat = 212
# x data length i.e. size of temporal dimension
Nx = 48

# x axis
x = np.array([x for x in range(Nx)])

# sample 2 dimensional y data randomly around a x**2 function
y = np.array([ 2*np.cosh(0.5*(x-Nx/2)) ]*N_dat,dtype=np.float)
y += np.random.randn(N_dat,Nx)

# truncate x for better fir range
x_red = x[7:41]

# perform uncorrelated fit from qcdanalysistools.fitting
Fit = tools.fitting.LeastSquare(fit_model, t_ydata=y[:,7:41], t_xdata=x_red)
result = Fit.fit()

# print results and compare by hand
print("My fit Stats: ============================================")
for key in result:
    print(key +": ", result[key])

# plot the fit to see if it actually worked
plt.plot(x,np.average(y,axis=0),'o',label="Raw Data")
plt.plot(x_red,result['Best fit'], label="LeastSquare")

plt.legend()
plt.grid()
plt.show()
