# HMC

A few simple examples that walk through a Hybrid/Hamiltonian Monte Carlo (HMC) Implementation.
Is an illustrative demonstration, to get a feel for Hamiltonian Dynamics and see how it can be used for sampling from the posterior.

## hamiltonian_dynamics_example.py

Follows from some of the examples from [1, Fig. 1]. For a 1-D (one position variable and one auxillary momentum variable), evaluates and plots the dynamics. Illustrates effects of different step sizes in leapfrog method.

## hmc_gaussian.py

A toy problem that uses HMC to sample from posterior. Is a simple 2-d example, where we are interested in the posterior over the mean (assuming known covariance in lieklihood). Ie., using a conjugate Gaussian model,

```p(q) = prior = N(q|mu_0, Sigma_0)```

```p(x|q) = Likelihood = N(x|q, Sigma_L)```

```p(q|x) = poterior, proportional to p(q)p(x|q)```


Conjugate model so can compare analytical solution with HMC samples to see if we are getting there.
