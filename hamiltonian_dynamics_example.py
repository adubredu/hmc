""" Demo to try and get HMC working 

Author: Ethan Goan
        ej.goan@qut.edu.au
Licence: Apache Licence 2.0 

Notation: Following examples and similar notation
to that in [1], which is probably not the best choice of
variable names, but have chosen param names here to remain
consistent with this.

q = params = position (posterior params of interest)
p = momentum/velocity
z = (q, p)
U(q) = Potential Energy
K(p) = Kinetic Energy
H(q, p) = Hamiltonian 
        = Total Energy
        = U(q) + K(p)
  
Derivatives: (assume partial derivatives)
dq_i/dt = dH/dp_i
dp_i/dt = - dH/dq_i
dz/dt = J x gradient(H(z))
where J = [ 0_{dxd}   I_{dxd} ]
          [-I_{dxd}   0_{dxd} ]
is a matrix and d is dimension of q

  
[1] Neal, Radford. "MCMC using Hamiltonian dynamics."
    Handbook of Markov Chain Monte Carlo (2011).
"""

import os
import sys

import numpy as np
from scipy.stats import multivariate_normal
import scipy
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


def one_d(n_samp = 20):
  """ simple 1-d example from Neal Sec 2.
  Will try and reproduce Fig. 1
  
  Corresponds to a standard Normal N(0,1)
  U(q) = q^2 / 2
  K(m) = p^2 / 2
  M = 1, so we don't need to worry about the mass here
  
  the true solution is of the form,
  q(t) = r * cos(a + t)
  p(t) = -r * sin(a + t)
  """
  # will run tests for different leap from sizes to compare
  epsilon = np.array([0.1, 0.15, 0.3, 1.3])
  tmp = np.linspace(-1.0, 1.0, 50) # used to plot true solution
  # initialise position and momentum
  q = np.zeros([4,n_samp])
  p = np.zeros([4,n_samp])
  # set intial momentum to 1
  p[:,0] = 1.0

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
  for j in range(0, len(epsilon)):
    axs = fig.add_subplot(2,2, j + 1)
    for i in range(1, n_samp):
      p[0, i] = p[0, i - 1] - epsilon[j] / 2.0 * q[0, i - 1] # half update
      q[0, i] = q[0, i - 1] + epsilon[j] * p[0, i]           # full update
      p[0, i] = p[0, i] - epsilon[j] / 2.0 * q[0, i]         # second half update
    axs.scatter(q, p)
    axs.plot(tmp, np.sqrt(1.0 - np.square(tmp)), c='gray', alpha=0.4)
    axs.plot(tmp, - np.sqrt(1.0 - np.square(tmp)), c='gray', alpha=0.4)
    axs.set_xlim([-1.6, 1.6])
    axs.set_ylim([-1.6, 1.6])
    axs.set_title('$\epsilon$ = {}'.format(epsilon[j]), usetex=True)
    axs.tick_params(axis = 'both', width = 0.2)
  ax.set_ylabel('Momentum $(p)$')
  ax.set_xlabel('Position $(q)$', usetex=True)
  plt.show()  
  


if __name__ == "__main__":
  one_d()
