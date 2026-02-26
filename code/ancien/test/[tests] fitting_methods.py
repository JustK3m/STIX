"""
You will find here all models tested with random data. This file serves as a test to see if the fitting in do_fit.py can
be done or if the model in question requires adaptation.
"""

from astropy.modeling import models, fitting
# from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

fitter = fitting.LevMarLSQFitter()

n = 100
noise = 0.1

x = np.linspace(-30, 30, n)
y = np.zeros(n)
y_err = np.ones(n)

"""
print("==================================== Power Law ======================================")
plt.figure(1)
plt.title("Power Law")

amplitude, x0, alpha = 1, 1, 1
xpl = np.linspace(1, 5, n)[:]
ypl = amplitude * (xpl / x0) ** (-alpha)
ypl = np.array([y_point + np.random.normal(0, noise) for y_point in ypl])[:]
index = ~(np.isnan(xpl) | np.isnan(ypl))
print("xpl = ", xpl)
print("\nypl = ", ypl)
plt.plot(xpl, ypl, color="blue", label="Original data")

ypl_err = y_err
model = models.PowerLaw1D()
print("\nModel : ", model)
print("\nIndex = ", index)
best_fit = fitter(model, xpl[index], ypl[index])  # , weights=1.0/ypl_err**2)
plt.plot(xpl, best_fit(xpl), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")

print("================================ Broken Power Law ===================================")
plt.figure(2)
plt.title("Broken Power Law")

amplitude, xbreak, alpha1, alpha2 = 1, 0, 1, 2
xbpl = xpl[:]
ybpl = amplitude * (x/xbreak) ** (-alpha1) if x < xbreak else amplitude * (x/xbreak) ** (-alpha2)
ybpl = np.array([y_point + np.random.normal(0, noise) for y_point in ybpl])[:]
plt.plot(xbpl, ybpl, color="blue", label="Original data")

ybpl_err = y_err
model = models.BrokenPowerLaw1D()
best_fit = fitter(model, xbpl, ybpl, weights=1.0/ybpl_err**2)
plt.plot(xbpl, best_fit(xbpl), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")
"""
print("==================================== Gaussian =======================================")
plt.figure(3)
plt.title("Gaussian")

mu, sigma, amplitude = 0.1, 1, 1  # 1000, 6.7, 0.1

xg = x[:]
yg = amplitude * np.exp(-(xg-mu)**2 / (2*sigma**2))
yg = np.array([y_point + np.random.normal(0, 0.1 * noise) for y_point in yg])[:]
plt.plot(xg, yg, color="blue", label="Original data")

yg_err = np.ones(n) * sigma
model = models.Gaussian1D()
best_fit = fitter(model, xg, yg, weights=1.0/yg_err**2)
plt.plot(xg, best_fit(xg), color="red", label="Fitted data")
plt.xlim(-5, 5)
print(best_fit)

plt.legend()
print("\n")

print("=================================== Polynomial ======================================")
plt.figure(4)
plt.title("Polynomial")

c0, c1, c2, c3 = 0.49486378, -0.05841062, 0.00177594, 1
xp = x[:]
yp = c3 ** xp**3 + c2 * xp**2 + c1 * xp + c0
yp = np.array([y_point + np.random.normal(0, 100 * noise) for y_point in yp])[:]
plt.plot(xp, yp, color="blue", label="Original data")

yp_err = y_err
model = models.Polynomial1D(degree=3)
best_fit = fitter(model, xp, yp, weights=1.0/yp_err**2)
plt.plot(xp, best_fit(xp), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")

print("=================================== Exponential =====================================")
plt.figure(5)
plt.title("Exponential")

amplitude, tau = 1, 1
xe = np.linspace(-5, 4, n)[:]
ye = -amplitude * np.exp(xe/tau)
ye = np.array([y_point + np.random.normal(0, 20 * noise) for y_point in ye])[:]
plt.plot(xe, ye, color="blue", label="Original data")

ye_err = y_err
model = models.Exponential1D()
best_fit = fitter(model, xe, ye, weights=1.0/ye_err**2)
plt.plot(xe, best_fit(xe), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")
"""
print("====================== Single Power Law Times an Exponential ========================")
plt.figure(6)
plt.title("Single Power Law Times an Exponential")

amplitude, x0, alpha, xcutoff = 1, 0, 1, 0
xspl = np.linspace(1, 5, n)[:]
yspl = amplitude * (x/x0)**(-alpha) * np.exp(-x/xcutoff)
yspl = np.array([y_point + np.random.normal(0, noise) for y_point in yspl])[:]
plt.plot(xspl, yspl, color="blue", label="Original data")

yspl_err = y_err
model = models.ExponentialCutoffPowerLaw1D()
best_fit = fitter(model, xspl, yspl, weights=1.0/yspl_err**2)
plt.plot(xspl, best_fit(xspl), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")
""""""
print("=============================== Logistic regression =================================")
plt.figure(7)
plt.title("Logistic Regression")

mu, s = 0, 10
xlr = x[:]
ylr = 1 / (1 + np.exp((mu - x)/s))
ylr = np.array([y_point + np.random.normal(0, noise) for y_point in ylr])[:]
plt.plot(xlr, ylr, color="blue", label="Original data")

ylr_err = y_err
model = LogisticRegression()
xlr = xlr.reshape(1, -1).T
print("xlr : ", xlr.shape)
print("ylr : ", ylr.shape)
best_fit = model.fit(xlr, ylr)
# best_fit = fitter(model, xlr, ylr, weights=1.0/ylr_err**2)
plt.plot(xlr, best_fit(xlr), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")
""""""
print("===================================== Lorentz =======================================")
plt.figure(8)
plt.title("Lorentz")

amplitude, gamma, x0 = 2, 1, 0
xlz = x[:]
ylz = (amplitude / np.pi) * (0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2)
ylz = np.array([y_point + np.random.normal(0, noise) for y_point in ylz])[:]
plt.plot(xlz, ylz, color="blue", label="Original data")

ylz_err = y_err
model = models.Lorentz1D()
best_fit = fitter(model, xlz, ylz, weights=1.0/ylz_err**2)
plt.plot(xlz, best_fit(xlz), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")
""""""
print("===================================== Moffat ========================================")
plt.figure(9)
plt.title("Moffat")

amplitude, x0, gamma, alpha = 1, 0, 1, 1
xmf = x[:]
ymf = amplitude * (1 + (x - x0)**2 / gamma**2) ** (-alpha)
ymf = np.array([y_point + np.random.normal(0, noise) for y_point in ymf])[:]
plt.plot(xmf, ymf, color="blue", label="Original data")

ymf_err = y_err
model = models.Moffat1D()
best_fit = fitter(model, xmf, ymf, weights=1.0/ymf_err**2)
plt.plot(xmf, best_fit(xmf), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")
""""""
print("====================================== Voigt ========================================")
plt.figure(10)
plt.title("Voigt")

gamma, sigma = 1, 1
xvgt = x[:]
yvgt = np.exp(-gamma * np.abs(x) - sigma**2 * x**2 / 2)
yvgt = np.array([y_point + np.random.normal(0, noise) for y_point in yvgt])[:]
plt.plot(xvgt, yvgt, color="blue", label="Original data")

yvgt_err = y_err
model = models.Voigt1D()
best_fit = fitter(model, xvgt, yvgt, weights=1.0/yvgt_err**2)
plt.plot(xvgt, best_fit(xvgt), color="red", label="Fitted data")
print(best_fit)

plt.legend()
print("\n")
""""""

# Feel free to edit this one at your convenience.
print("=================================== Multi-Test ======================================")
plt.figure(11)
plt.title("Multi-Test")

plt.plot(xe, ye, color="blue", label="Exp")

model = models.Polynomial1D(degree=2)
x2 = np.linspace(2, 4, n)[:]
y2 = -amplitude * np.exp(x2/tau)
y2 = np.array([y_point + np.random.normal(0, 20 * noise) for y_point in y2])[:]
best_fit = fitter(model, x2, y2, weights=1.0/ye_err**2)
plt.plot(x2, best_fit(x2), color="red", label="Fit gauss")

plt.legend()
print("\n")
"""
plt.show()
