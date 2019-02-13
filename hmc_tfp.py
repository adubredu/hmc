import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def MVN_data(mean, cov, n_samp):
  return np.random.multivariate_normal(mean.flatten(), cov, size = n_samp).astype(np.float64)


def MVN_prior(size):
  return tfd.MultivariateNormalDiag(loc = np.zeros(size).astype(np.float64))

def MVN_likelihood(q, cov):
  return tfd.MultivariateNormalFullCovariance(loc = q, covariance_matrix = cov)

def potential(q, prior, data):
  likelihood = MVN_likelihood(q, cov)
  return (prior.log_prob(q) + tf.reduce_mean(likelihood.log_prob(data)))


def plot_results(q_prior, cov_array, x, q_pos):
  """Plot prior, data and our posterior"""

  # evaluate the true posterior
  x_bar = np.mean(x, axis = 0).reshape(-1, 1)
  print(x.shape)
  print(x_bar.shape)
  likelihood_prec = np.linalg.inv(cov_array)
  cov = np.linalg.inv( np.eye(2) + likelihood_prec)
  mean = cov @ (likelihood_prec @ x_bar + np.array([[0], [0]]))
  X = np.linspace(-2.5,2.5,500)
  Y = np.linspace(-2.5,2.5,500)
  X,Y = np.meshgrid(X,Y)
  pos = np.dstack([X, Y])
  
  # generate RV for true posterior and prior so we can evaluate and get contours
  true_post_rv = multivariate_normal(mean.flatten(), cov)
  prior_rv = multivariate_normal(np.zeros(2), np.eye(2))
  # stack all found positions in list into an array
  print(x.shape)
  print(q_pos)
  Q = np.vstack(q_pos)
  print(Q.shape)
  # plot prior, data and likelihood
  fig = plt.figure(figsize=(10,10))
  # add the global x and y labels
  ax = fig.add_subplot(1,1,1)
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
  ax.set_ylabel('$q_2$')
  ax.set_xlabel('$q_1$')
  # now add the proper plots
  ax1 = fig.add_subplot(131)
  ax1.set_title('Prior')
  ax1.set_xlim([-2.5, 2.5])
  ax1.set_ylim([-2.5, 2.5])
  contour = ax1.contour(X, Y, prior_rv.pdf(pos).reshape(500,500), 4)
  ax2 = fig.add_subplot(132)
  ax2.set_title('Data')
  ax2.set_xlim([-2.5, 2.5])
  ax2.set_ylim([-2.5, 2.5])
  ax2.scatter(x[:, 0], x[:, 1])
  ax3 = fig.add_subplot(133)
  ax3.set_title('Posterior')
  ax3.set_xlim([-2.5, 2.5])
  ax3.set_ylim([-2.5, 2.5])
  contour = ax3.contour(X, Y, true_post_rv.pdf(pos).reshape(500,500), 4)
  ax3.clabel(contour, inline=1, fontsize=10)
  ax3.scatter(Q[:, 0], Q[:, 1], alpha = 0.2, label = 'samples')
  ax3.scatter(Q[0, 0], Q[0, 1], c='g', label = 'start pos.')
  ax3.scatter(Q[-1, 0], Q[-1, 1], c='r', label = 'end pos.')
  ax3.legend()
  plt.show()
  


if __name__ == '__main__':
  # our generative model
  cov_array = np.array([[1.0, 0.9], [0.9, 1.0]]).astype(np.float64)
  cov = tf.constant(cov_array, dtype=tf.float64)
  prior = MVN_prior(2)
  data = MVN_data(np.array([1.0, 2.0]), cov_array, 100)

  # Create state to hold updated `step_size`.
  step_size = tf.get_variable(
    name='step_size',
    initializer=np.float64(0.1),
    use_resource=True,  # For TFE compatibility.
    trainable=False)

  joint_log_prob = lambda q: potential(q, prior, data)
  # Initialize the HMC transition kernel.
  num_results = int(1e3)
  num_burnin_steps = int(1e3)
  print(data.shape)
  hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=joint_log_prob,
    num_leapfrog_steps=10,
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

    print('mean:{}  stddev:{:.4f}  acceptance:{:.4f}'.format(
      tf.reduce_mean(samples_, axis = 0).eval(), samples_.std(), kernel_results_.is_accepted.mean()))
    samples = samples.eval()
    print(samples)

    plot_results(prior, cov_array, data, samples)

