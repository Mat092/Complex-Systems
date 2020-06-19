import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

def polyfit(x, y, grade=1):
  '''
  Polynomial fit of n points, given by x and y, by an assigned grade. Least Squared method

  Parameters:
    x     : numpy array of points in the x domain
    y     : numpy array of samples
    grade : int, maximum grade of the polinomial. (min 1)

  Return:
    coeffs : numpy array of coefficients of the polynomial fit
  '''

  if grade < 0:
    raise ValueError(f'grade is {grade}, but lesser than 0 is not accetable')

  n       = x.size
  x.shape = (n,1)
  X       = np.concatenate([np.power(x, i) for i in range(grade+1)], axis=1)

  coeffs = np.linalg.inv(X.T @ X) @ X.T @ y # that's (X_t * X)^-1 * X_t * y

  return coeffs, X

np.random.seed(123) # set the seed for RNG

grade    = 15
n        = 100    # number of sample points
function = np.sin

x = np.linspace(0, 10, n)   # equally spaced points in [0,10]
y = 5*function(0.9*x)       # function to fit, parameters are just for flavour

sample = y + np.random.normal(loc=0, scale=1, size=y.shape) # sample with gaussian noise μ = 0, σ^2 = 1

coeff, X = polyfit(x=x, y=sample, grade=grade) # computes the coefficients and the matrix X

y_fit = X @ coeff  # fit function.

# PLOT ----------------------------------------
fig, ax1 = plt.subplots(figsize=(10,7))

ax1.plot(x, sample, '.', label='Samples')
ax1.plot(x, y, '--', label='True function')
ax1.plot(x, y_fit, '-', label=f'Fit (order {grade})')
ax1.set_ylim(-6, 6)
ax1.legend()

sns.despine(trim=True)
plt.savefig('./images/fit_nlinear_order{}.png'.format(grade))
plt.show()
