import numpy as np
from scipy.stats import multivariate_normal
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def MVN_data(mean, cov, n_samp):
  return np.random.multivariate_normal(mean.flatten(), cov, size = n_samp).astype(np.float64)


def MVN_prior(size):
  return tfd.MultivariateNormalDiag(loc = np.zeros(size).astype(np.float64))

def potential(q):#q, prior, likelihood, data):
  def MVN_likelihood(q, cov):
    return tfd.MultivariateNormalFullCovariance(q, cov)
  likelihood = MVN_likelihood(q, cov)
  return -prior.log_prob(q) - tf.reduce_sum(likelihood.log_prob(data.astype(np.float64)), axis = 0)

#tf.enable_eager_execution()
# this is the variable we are trying to perform inference over
#q = tf.Variable(np.array([2.0, 1.0]), name='q',  dtype=tf.float64)
#print('q = {}'.format(q))

# our generative model
cov_array = np.array([[1.0, 0.9], [0.9, 1.0]]).astype(np.float64)
cov = tf.constant(cov_array, dtype=tf.float64)
prior = MVN_prior(2)
data = MVN_data(np.array([1.0, 2.0]), cov_array, 10)
print(type(data[0,0]))

print(data)
print(cov)
#print(likelihood)
print(prior)
#print(q)

# Create state to hold updated `step_size`.
step_size = tf.get_variable(
  name='step_size',
  initializer=np.float64(1.0),
  use_resource=True,  # For TFE compatibility.
  trainable=False)

# Initialize the HMC transition kernel.
num_results = int(10e3)
num_burnin_steps = int(1e3)
print(data.shape)
hmc = tfp.mcmc.HamiltonianMonteCarlo(
  target_log_prob_fn=potential,
  num_leapfrog_steps=3,
  step_size=step_size,
  step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
    num_adaptation_steps=int(num_burnin_steps * 0.8)))

# Run the chain (with burn-in).
samples, kernel_results = tfp.mcmc.sample_chain(
  num_results=num_results,
  num_burnin_steps=num_burnin_steps,
  current_state = np.array([0.0, 0.0]).astype(np.float64),
  kernel=hmc)

# Initialize all constructed variables.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  init_op.run()
  samples_, kernel_results_ = sess.run([samples, kernel_results])
  
  print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
    samples_.mean(), samples_.std(), kernel_results_.is_accepted.mean()))
    
