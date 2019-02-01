#!/usr/bin/env python
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
import tensorflow as tf
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt



def grad_potential_gaussian(q, prior, likelihood):
  """Computes gradient of potential energy for MVN

  Potential energy for HMC is typically defined as,
  U(q) = -log(p(q) p(x | q)) = -log(p(q)) - log(p(x | q))
  And we will assume that we have a Gaussian Prior and likelihood.
  From the Matrix Cookbook [1, equation 85], know that
  d(log(p(x)))/dq = Sigma^{-1}(q - mu)
  This is because we are using the log of the distribution, so we get rid
  of the exp() and seperate the exponent from the normalising constant,
  and when taking the derivative the constant terms will vanish, and
  because a valid covariance matrix is symmetric.
  
  Args:
    x (scipy.stats_multivariate_normal)
      variable that we want to find the gradient of
  """
  
  



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
  ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
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
  


def simple_gaussian_hmc(epsilon = 0.2, L = 10, iters = 100):
  """Compare HMC for sample where solution is known
  
  Plot HMC draws from posterior against that of true posterior
  to see how it all works.
  Will work with a 2D Gaussian, and will look at posterior over the 
  mean. Using same notation as above (and including new variables for
  covariance etc.),
  p(q) = N(q|0, I) = Prior 
  p(x | q) = N(x| q, C)
  p(q | x) = N(q| mu, Sigma) 
  Will maybe type this out in TeX for full derivation

  Args:
    epsilon (float):
      step-size in leapfrom algorithm
    L (int):
      Number of steps to take before updating
    iters (int):
      number of times to iterate through series of L steps
      (L steps per iteration)
  """
  # initial value of position
  current_q = np.zeros([0, 0])
  # prior with independant components
  q_prior = multivariate_normal([0. 0], mean = 0, cov = 1)
  # momentum
  p_dist = multivariate_normal([0. 0], mean = 0, cov = 1)
  # where we will save the accepted values of position
  q_acc = []
  # lets get to it
  for i in range(0, iters):
    q = current_q
    p = p_dis.rvs()
    # save the current value of p
    p_current = p
    for j in range(0, L):

      # half update of momentum
      p = p - epsilon * 


      
      
      
      
if __name__ == "__main__":
  # start with a simple 1-d example
  one_d()
  
