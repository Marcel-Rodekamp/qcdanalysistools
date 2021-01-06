import numpy as np
import matplotlib.pyplot as plt
import qcdanalysistools as tools
import scipy

fit_cuntion_1 = lambda x,A,B: B*x + A
fit_cuntion_2 = lambda x,A,B,C: C*x**2 + fit_cuntion_1(x,A,B)
fit_cuntion_3 = lambda x,A,B,C,D: D*x**3 + fit_cuntion_2(x,A,B,C)
fit_cuntion_4 = lambda x,A,B,C,D,E: E*x**4 + fit_cuntion_3(x,A,B,C,D)

N = 100

x = np.array([x for x in range(N)])
y = np.array( [np.square(x) for x in range(N)] )
y_noise = np.array([ np.random.random()/100 for x in range(N)])
y = y + y_noise

cons_report = f"Function report:\n"
cons_report+= f"Created y(x) = x\u00B2 + noise\n"
cons_report+= f"Fitting to, linear, quadratic, cubic, quartic polynomials\n"

popt_1,pcov_1 = scipy.optimize.curve_fit(fit_cuntion_1, xdata = x, ydata = y)
popt_2,pcov_2 = scipy.optimize.curve_fit(fit_cuntion_2, xdata = x, ydata = y)
popt_3,pcov_3 = scipy.optimize.curve_fit(fit_cuntion_3, xdata = x, ydata = y)
popt_4,pcov_4 = scipy.optimize.curve_fit(fit_cuntion_4, xdata = x, ydata = y)

fit_1 = fit_cuntion_1(x,*popt_1)
fit_2 = fit_cuntion_2(x,*popt_2)
fit_3 = fit_cuntion_3(x,*popt_3)
fit_4 = fit_cuntion_4(x,*popt_4)

chisq_1,_ = scipy.stats.chisquare(fit_1, f_exp = y)
chisq_2,_ = scipy.stats.chisquare(fit_2, f_exp = y)
chisq_3,_ = scipy.stats.chisquare(fit_3, f_exp = y)
chisq_4,_ = scipy.stats.chisquare(fit_4, f_exp = y)

chisq_report = f"\u03C7\u00B2 report:\n"
chisq_report+= f"linear : {chisq_1}\n"
chisq_report+= f"square : {chisq_2}\n"
chisq_report+= f"cubic  : {chisq_3}\n"
chisq_report+= f"quadric: {chisq_4}\n"

AIC_1 = tools.stats.AIC_chisq(2,chisq_1)
AIC_2 = tools.stats.AIC_chisq(3,chisq_2)
AIC_3 = tools.stats.AIC_chisq(4,chisq_3)
AIC_4 = tools.stats.AIC_chisq(5,chisq_4)

AIC_weights = tools.stats.AIC_weights(np.array([AIC_1,AIC_2,AIC_3,AIC_4]))

AIC_report = f"AIC report:\n"
AIC_report+= f"linear : AIC = {AIC_1}, w[AIC] = {AIC_weights[0]}\n"
AIC_report+= f"square : AIC = {AIC_2}, w[AIC] = {AIC_weights[1]}\n"
AIC_report+= f"cubic  : AIC = {AIC_3}, w[AIC] = {AIC_weights[2]}\n"
AIC_report+= f"quadric: AIC = {AIC_4}, w[AIC] = {AIC_weights[3]}\n"

AICc_1 = tools.stats.AICc_chisq(2,N,chisq_1)
AICc_2 = tools.stats.AICc_chisq(3,N,chisq_2)
AICc_3 = tools.stats.AICc_chisq(4,N,chisq_3)
AICc_4 = tools.stats.AICc_chisq(5,N,chisq_4)

AICc_weights = tools.stats.AIC_weights(np.array([AICc_1,AICc_2,AICc_3,AICc_4]))

AICc_report = f"AICc report:\n"
AICc_report+= f"linear : AICc = {AICc_1}, w[AICc] = {AICc_weights[0]}\n"
AICc_report+= f"square : AICc = {AICc_2}, w[AICc] = {AICc_weights[1]}\n"
AICc_report+= f"cubic  : AICc = {AICc_3}, w[AICc] = {AICc_weights[2]}\n"
AICc_report+= f"quadric: AICc = {AICc_4}, w[AICc] = {AICc_weights[3]}\n"

print(cons_report)
print(chisq_report)
print(AIC_report)
print(AICc_report)

plt.plot(y,'+', label="Raw Dat")
plt.plot(fit_1, label="linear")
plt.plot(fit_2, label="square")
plt.plot(fit_3, label="cubic ")
plt.plot(fit_4, label="quadric")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
