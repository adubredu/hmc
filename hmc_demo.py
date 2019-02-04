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
import scipy
#import tensorflow as tf
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt




class gaussian(object):
  def __init__(self, mean = 0, cov = 0):
    self.mean = np.array(mean).reshape(-1, 1)
    self.cov = cov
    self.dim = self.mean.shape[0]
    self.prec = np.linalg.inv(cov)
    # check that that covariance is SPD
    self.Z = 1.0 / np.sqrt((2 * np.pi) ** self.dim * np.linalg.det(self.cov))
    # check that the dimensions of all the values
    # are consistent
    if(self.mean.shape[0] != self.cov.shape[0]):
      raise(ValueError('Inconsistent dimensions for mean and cov'))
    if(self.cov.shape[0] != self.cov.shape[1]):
      raise(ValueError('Covariance Matrix should be square'))        

  def sample(self):
    print(self.mean.shape)
    return np.random.multivariate_normal(self.mean.reshape(self.dim), self.cov).reshape(-1, 1)

  def eval(self, x):
    return self.Z * np.exp(-0.5 * (
      (x - self.mean).T @ self.prec @ (x - self.mean)))
  

def grad_potential_gaussian(q, x, prior, cov_rep):
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
    x (Np array)
      data drawn from the likelihood. Each column is an independent
      draw from likelihood (column = single sample)
  """
  # now compute the gradient
  # assumed a standard normal prior so won't bother inverting
  # $I$ matrix for covariance
  # also assumed zero mean so wont bother subtracting away
  print('q = {}'.format(q))
  print(q.shape)
  print(x.shape)
  dq_prior = np.matmul(prior.cov, q)
  x_rep = (x - q).reshape(-1, 1)
  print('x_rep shape = {}'.format(x_rep.shape))
  dq_likelihood = np.sum(np.matmul(cov_rep, x_rep), axis = 0)
  return(dq_prior + dq_likelihood)


def potential(q, x, prior, cov_rep, cov_det, n_samp):
  x_rep = (x - q).reshape(-1, 1)
  print('x_rep.shape = {}'.format(x_rep.shape))
  print('q.shape = {}'.format(q.shape))
  print('cov.shape = {}'.format(cov_rep.shape))
  print('cov_det.shape = {}'.format(cov_det.shape))
  U_prior = np.log(prior.eval(q))
  print(U_prior)
  U_likelihood = (-n_samp / 2) * (n_samp * np.log((2 * np.pi)) + np.log(cov_det)) - 0.5 * np.sum(np.matmul(np.matmul(x_rep.T, cov_rep), x_rep), axis = 0)
  print(U_likelihood)
  return(U_prior + U_likelihood)



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
  


def simple_gaussian_hmc(epsilon = 0.002, L = 10, iters = 100, n_samp = 100):
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
  q_prior = gaussian(mean = [0, 0], cov = np.eye(2))
  # draw psuedo-data from the likelihood
  cov = np.array([[1.0, 2.0], [2.0, 4.0]]) + np.eye(2)
  cov_det = np.linalg.det(cov)
  print('cov_det = {}'.format(cov_det))
  x = np.random.multivariate_normal(np.array([1, 2]),
                                     cov, size = n_samp).reshape(2, n_samp)
  # repeating covariance where all samples are independent
  cov_rep = scipy.linalg.block_diag(np.kron(np.eye(n_samp), cov))
  # where we will save the accepted values of position
  q_acc = []
  p_dist = gaussian([0, 0], np.eye(2))
  # lets get to it
  for i in range(0, iters):
    q = current_q
    p = p_dist.sample()
    # save the current value of p
    print('p = {}'.format(p))
    current_p = p
    #print('q = {}'.format(q))
    # half update of momentum
    p = p - epsilon * grad_potential_gaussian(q, x, q_prior, cov_rep) / 2.0
    for j in range(0, L):
      # full update of the position
      print('before update q = {}'.format(q))
      print('p = {}'.format(p))
      print('p.shape = {}'.format(p.shape))
      q = q + epsilon * p
      print('after update q = {}'.format(q))
      # make a full step in momentum unless we are on the last step
      if(j < L -1):
        p = p - epsilon * grad_potential_gaussian(q, x, q_prior, cov_rep)

    # make a half step and then negate the momentum term
    p = p - epsilon * grad_potential_gaussian(q, x, q_prior, cov_rep) / 2.0
    p = -p

    # evaluate the potential and kinetic energies to see if we accept or reject
    current_U = potential(current_q, x, q_prior, cov_rep, cov_det, n_samp)
    current_K = np.sum(current_p**2 / 2.0)
    proposed_U = potential(q, x, q_prior, cov_rep, cov_det, n_samp)
    proposed_K = np.sum(p**2 / 2.0)
    print(current_U)
    # now see if we accept or reject
    if(1.0 < np.exp(current_U - proposed_U + current_K - proposed_K)):
      # then we are accepting so save it and set the new current q value
      q_pos.append(q)
      current_q = q 

  print('found q = {}'.format(q_pos))
      
      
      
      
if __name__ == "__main__":
  # start with a simple 1-d example
  #one_d()
  simple_gaussian_hmc()
