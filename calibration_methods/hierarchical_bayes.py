import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import torch
from torch.nn.functional import softmax

""" This file implements the Hierarchical Bayesian calibrator. 
"""

class HierarchicalBayesianCalibrator:

    def __init__(self, prior_params, num_classes, **kwargs):
        self.num_classes = num_classes
        # Inference parameters
        self.NUTS_params = {'adapt_step_size': kwargs.pop('adapt_step_size', True),
                            'target_accept_prob': kwargs.pop('target_accept_prob', 0.8),
                            'max_plate_nesting': 1
                            }
        self.mcmc_params = {'num_samples': kwargs.pop('num_samples', 250),
                            'warmup_steps': kwargs.pop('num_warmup', 1000),
                            'num_chains': kwargs.pop('num_chains', 4)
                            }

        # Constraints on delta; choices are None, 'soft', 'hard'
        self.delta_constraint = kwargs.pop('delta_constraint', 'soft')
        assert self.delta_constraint in [None, 'soft', 'hard'], 'Invalid delta constraint'

        # Prior parameters on beta / delta ; assumes each delta is iid
        self.prior_params = {'mu_beta': prior_params['mu_beta'],
                             'sigma_beta': prior_params['sigma_beta'],
                             'mu_delta': torch.empty(self.num_classes).fill_(prior_params['mu_delta']),
                             'sigma_delta': torch.empty(self.num_classes).fill_(prior_params['sigma_delta'])}

        # Tracking params
        # TODO: Prior/posterior trace
        self.timestep = 0
        self.mcmc = None  # Contains the most recent Pyro MCMC api object

        print('\nInitializing HBC model:\n'
              '----| Prior: {} \n----| Inference Method: NUTS \n'
              '----| MCMC parameters: {}'
              ''.format(prior_params, self.mcmc_params))

    def update(self, logits, labels):
        """ Performs an update given new observations.

        Args:
            logits: tensor ; shape (batch_size, num_classes)
            labels: tensor ; shape (batch_size, )
        """
        assert len(labels.shape) == 1, 'Got label tensor with shape {} -- labels must be dense'.format(labels.shape)
        assert len(logits.shape) == 2, 'Got logit tensor with shape {}'.format(logits.shape)
        assert (labels.shape[0] == logits.shape[0]), 'Shape mismatch between logits ({}) and labels ({})' \
            .format(logits.shape[0], labels.shape[0])

        print('delta constraint::')
        print(self.delta_constraint)
        self.timestep += 1

        logits = logits.detach().clone().requires_grad_()
        labels = labels.detach().clone()

        batch_size = labels.shape[0]
        print('----| Updating HBC model\n--------| Got a batch size of: {}'.format(batch_size))

        # TODO: Update prior (for sequential)
        # self._update_prior()
        # print('--------| Updated priors: {}'.format(self.prior_params))

        print('--------| Running inference ')
        nuts_kernel = NUTS(hbc_model, **self.NUTS_params)
        self.mcmc = MCMC(nuts_kernel, **self.mcmc_params, disable_progbar=False)
        print('.')
        self.mcmc.run(self.prior_params, logits, labels, self.delta_constraint)
        print('..')

        #  TODO: update posterior (for sequential)
        # self._update_posterior(posterior_samples)

        return self.mcmc

    def get_current_posterior_samples(self):
        """ Returns the current posterior samples for beta, delta, and alpha.
        """
        if self.mcmc is None:
            return None

        posterior_samples = self.mcmc.get_samples()
        posterior_samples['alpha'] = posterior_samples['beta'].view(-1, 1) + posterior_samples['delta']

        return posterior_samples

    def calibrate(self, logit):
        """ Calibrates the given batch of logits using the current posterior samples.

        Args:
            logit: tensor ; shape (batch_size, num_classes)
        """
        # Get alpha samples
        alpha_samples = self.get_current_posterior_samples()['alpha']  # Shape (num_samples, num_classes)
        num_samples, num_classes = alpha_samples.shape

        # Map alphas to temperatures
        temperature_samples = torch.exp(-1. * alpha_samples)  # Shape (num_samples, num_classes)

        # Get a batch of logits for each sampled temperature
        # Shape (num_samples, batch_size, num_classes)
        tempered_logit_samples = temperature_samples.view(num_samples, 1, num_classes) * logit

        # Softmax the sampled logits to get sampled probabilities
        prob_samples = softmax(tempered_logit_samples, dim=2)  # Shape (num_samples, batch_size, num_classes)

        # Average over the sampled probabilities to get Monte Carlo estimate
        calibrated_probs = prob_samples.mean(dim=0)   # Shape (batch_size, num_classes)

        return calibrated_probs


# NB: Labels must be in [0, 1, 2, . . . num_classes - 1] !
def hbc_model(prior_params, logits, labels, delta_constraint=None):
    """ This function defines the Pyro model for HBC.
    """
    n_obs = logits.shape[0]  # Batch size
    n_classes = logits.shape[1]  # Number of classes

    # Prior over global temperature Beta ~ N( beta_mu, beta_sigma^2 )
    prior_beta_mu = prior_params['mu_beta']
    prior_beta_sigma = prior_params['sigma_beta']
    beta = pyro.sample('beta', dist.Normal(prior_beta_mu, prior_beta_sigma))  # Shape (1, )

    # Prior over delta's ; vectorized
    prior_delta_mu = prior_params['mu_delta']  # Shape (n_classes, )
    prior_delta_sigma = prior_params['sigma_delta']  # Shape (n_classes, )
    delta = pyro.sample('delta', dist.Normal(prior_delta_mu, prior_delta_sigma))  # Shape (n_classes, )
    # Sum-to-zero constraints on delta
    if delta_constraint == 'hard':
        delta = delta - delta.mean()
    elif delta_constraint == 'soft':
        eps = 0.01  # Could be tuned
        pyro.sample('sum(delta)', dist.Normal(0, eps * n_classes), obs=delta.sum())

    alpha = beta + delta  # Shape (n_classes, ) ; deterministic given beta, delta
    probs = softmax(torch.exp(-1. * alpha) * logits, dim=1)  # Shape (n_obs, n_classes) ; tempered probabilities

    # Observation plate ; vectorized
    with pyro.plate('obs', size=n_obs):
        a = pyro.sample('cat_obs', dist.Categorical(probs=probs), obs=labels)
