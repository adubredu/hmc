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

TODO - check that the covariance for all my likelihood references
       is set correctly (should be independent)
     - use Gaussian class for likelihood to make life easier
     - test some more
"""


import os
import sys

import numpy as np
from scipy.stats import multivariate_normal
import scipy
#import tensorflow as tf
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class Gaussian(object):
  """Class for Multivariate Gaussian"""
  def __init__(self, mean = 0.0, cov = 0.0):
    self.mean = np.array(mean).reshape(-1, 1).astype(np.float64)
    self.cov = cov.astype(np.float64)
    self.dim = self.mean.shape[0]
    self.prec = np.linalg.inv(self.cov)
    self.cov_det = np.linalg.det(self.cov)
    # check that that covariance is SPD
    self.Z = 1.0 / np.sqrt((2.0 * np.pi) ** self.dim * self.cov_det)
    # check that the dimensions of all the values
    # are consistent
    if(self.mean.shape[0] != self.cov.shape[0]):
      raise(ValueError('Inconsistent dimensions for mean and cov'))
    if(self.cov.shape[0] != self.cov.shape[1]):
      raise(ValueError('Covariance Matrix should be square'))        

  def sample(self, n_samp = 1):
    return np.random.multivariate_normal(self.mean.reshape(self.dim),
                                         self.cov,
                                         size = n_samp).T
  def eval_pdf(self, x):
    return self.Z * np.exp(-0.5 * (
      (x - self.mean).T @ self.prec @ (x - self.mean)))
  

def eval_true_post(prior, likelihood, x):
  """Will evaluate true posterior for Gaussian Prior/Posterior
  
  For Gaussian Prior N(q|mu_0, sigma_0)
      Gaussian Likelihood N(x|mu_l, sigma_l)
  From [1, eqn. 4.124 and 4.125]
  Posterior is,
  p(q|x) = N(q | mu, sigma)
  Sigma^{-1} = Sigma_0^{-1} + Sigma_l^{-1}
  mu = Sigma (Sigma_l^{-1}x + Sigma_0^{-1}mu_0)
  (To use result from [1], matrix A is set to identity and 
  vector b is set to zero).

  Args:
    prior (Gaussian):
      Gaussian Object
    likelihood (Gaussian):
      Gaussian Object    
    x (array):
      our data, with each column being an independent sample
  
  Returns:
    Gaussian Object with our true posterior
  
  References:
  [1] Murphy, K. (2012). "Machine Learning : A Probabilistic Perspective." 
      Cambridge: MIT Press.
  """
  
  x_bar = np.mean(x, axis = 1).reshape(-1, 1)
  #print(x_bar.shape)
  cov = np.linalg.inv(prior.prec + likelihood.prec)
  mean = cov @ (likelihood.prec @ x_bar + prior.prec @ prior.mean)
  return Gaussian(mean, cov)


def grad_potential_gaussian(q, x, prior, likelihood):
  """Computes gradient of potential energy for MVN

  Potential energy for HMC is typically defined as,
  U(q) = -log(p(q) p(x | q)) = -log(p(q)) - log(p(x | q))
  And we will assume that we have a Gaussian Prior and likelihood.
  From the Matrix Cookbook [1, eqn. 85], know that
  d(log(p(x)))/dq = Sigma^{-1}(q - mu)
  (85) applies to our prior, and equation (86) applies for the likelihood.
  They only differ by a negative sign.
  This is because we are using the log of the distribution, so we get rid
  of the exp() and seperate the exponent from the normalising constant,
  and when taking the derivative the constant terms will vanish, and
  because a valid covariance matrix is symmetric.
  
  Args:
    x (Np array)
      data drawn from the likelihood. Each column is an independent
      draw from likelihood (column = single sample)

  References:
  [1] Petersen, Kaare Brandt, and Michael Syskind Pedersen. 
  "The matrix cookbook." Technical University of Denmark 7.15 (2012): 510.
  """
  # now compute the gradient
  dq_prior = - np.matmul(prior.prec, (q - prior.mean).reshape(-1, 1))
  dq_likelihood =  np.matmul(likelihood.prec, np.mean(x - q, axis = 1).reshape(-1, 1))
  return -(dq_prior + dq_likelihood)


def potential(q, x, prior, likelihood, n_samp):
  "Calculate the potential energy at current position"
  U_log_prior = np.log(prior.eval_pdf(q))
  x_sum = np.mean(x - q, axis = 1).reshape(-1, 1)
  k = x.shape[0] 
  U_log_likelihood = - 0.5 * x_sum.T @ likelihood.prec @ x_sum - n_samp * np.log(likelihood.Z)
  return -(U_log_prior + U_log_likelihood)



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
  


def simple_gaussian_hmc(epsilon = 0.2, L = 100, iters = 100, n_samp = 100):
  """Demo to compare HMC for sample where solution is known
  
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
    n_samp (int):
      number of psuedo-data samples to work with
  """
  # initial value of position
  current_q = np.zeros(2).reshape(2,1)
  q_pos = []
  # prior with independant components
  q_prior = Gaussian(mean = [0.0, 0.0], cov = np.eye(2))
  # draw psuedo-data from the likelihood
  x_mu = np.array([0.5, 0.0])
  cov = np.array([[1.0, 0.9], [0.9, 1.0]])
  x_dist = Gaussian(mean = x_mu, cov = cov)
  x = x_dist.sample(n_samp)
  likelihood = Gaussian(mean = np.zeros(2), cov = cov)
  x = likelihood.sample(n_samp)
  # distribution for the momentum variable (standard normal)
  p_dist = Gaussian([0.0, 0.0], np.eye(2))
  # where we will save the accepted values of position
  q_acc = []
  # lets get too it
  for i in range(0, iters):
    q = current_q
    p = p_dist.sample()
    # save the current value of p
    current_p = p
    # half update of momentum
    p = p - epsilon * grad_potential_gaussian(q, x, q_prior, likelihood) / 2.0
    for j in range(0, L - 1):
      # full update of the position
      q = q + epsilon * p
      # make a full step in momentum unless we are on the last step
      p = p - epsilon * grad_potential_gaussian(q, x, q_prior, likelihood)
      
    # make a half step and then negate the momentum term
    p = p - epsilon * grad_potential_gaussian(q, x, q_prior, likelihood) / 2.0
    p = -p

    # evaluate the potential and kinetic energies to see if we accept or reject
    current_U = potential(current_q, x, q_prior, likelihood, n_samp)
    current_K = np.sum(current_p**2.0 / 2.0)
    proposed_U = potential(q, x, q_prior, likelihood, n_samp)
    proposed_K = np.sum(p**2.0 / 2.0)
    print('current_U = {}'.format(current_U))
    print('proposed_U = {}'.format(proposed_U))
    print('current_K = {}'.format(current_K))
    print('proposed_K = {}'.format(proposed_K))
    # now see if we accept or reject
    test = np.exp(current_U - proposed_U + current_K - proposed_K)
    one_samp = np.random.uniform(low = 0.0, high = 1.0)
    print('test = {}'.format(test))
    print('one_samp = {}'.format(one_samp))
    if(one_samp < test):
      # then we are accepting so save it and set the new current q value
      q_pos.append(q)
      current_q = q 
      
  # done sampling, lets print and plot our results
  accept_ratio = len(q_pos) / iters
  print('Accept Ratio = {}'.format(accept_ratio))
  true_post = eval_true_post(q_prior, likelihood, x)
  X = np.linspace(-1.5,1.5,500)
  Y = np.linspace(-1.5,1.5,500)
  X,Y = np.meshgrid(X,Y)
  pos = np.dstack([X, Y])
  rv = multivariate_normal(true_post.mean.flatten(), true_post.cov)
  Q = np.hstack(q_pos)
  fig = plt.figure(figsize=(10,10))
  ax0 = fig.add_subplot(111)
  contour = ax0.contour(X, Y, rv.pdf(pos).reshape(500,500), 4)
  ax0.clabel(contour, inline=1, fontsize=10)
  ax0.scatter(Q[0, :], Q[1, :], alpha = 0.2)
  ax0.scatter(Q[0, 0], Q[1, 0], c='g')
  ax0.scatter(Q[0, -1], Q[1, -1], c='r')
  #ax0.scatter(x[0, :], x[1, :], c='y')
  plt.show()

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
  ax.set_xlabel('X axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('Z axis')
  plt.show()

  
           

if __name__ == "__main__":
  # start with a simple 1-d example
  #one_d()
  simple_gaussian_hmc()
