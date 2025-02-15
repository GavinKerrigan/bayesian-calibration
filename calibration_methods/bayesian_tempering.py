import warnings
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import numpy as np
import torch
from torch.nn.functional import softmax

""" This class implements the Bayesian Tempering calibrator. 
"""


class BayesianTemperingCalibrator:
    """ This class implements the Bayesian tempering calibrator.
    Performs inference using NUTS.
    """

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

        # Prior parameters on beta / delta ; assumes each delta is iid
        self.prior_params = {'mu_beta': prior_params['mu_beta'],
                             'sigma_beta': prior_params['sigma_beta']}

        # Posterior parameters after ADF
        self.posterior_params = {'mu_beta': None,
                                 'sigma_beta': None}

        # Drift parameters for sequential updating
        self.sigma_drift = kwargs.pop('sigma_drift', 0.0)

        # Tracking params
        # TODO: Prior/posterior trace
        self.timestep = 0
        self.mcmc = None  # Contains the most recent Pyro MCMC api object
        self.verbose = kwargs.pop('verbose', True)

        if self.verbose:
            print('\nInitializing BT model:\n'
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

        logits = logits.detach().clone().requires_grad_()
        labels = labels.detach().clone()

        batch_size = labels.shape[0]
        if self.verbose:
            print('----| Updating HBC model\n--------| Got a batch size of: {}'.format(batch_size))

        self._update_prior_params()
        if self.verbose:
            print('--------| Updated priors: {}'.format(self.prior_params))
            print('--------| Running inference ')
        nuts_kernel = NUTS(bt_model, **self.NUTS_params)
        self.mcmc = MCMC(nuts_kernel, **self.mcmc_params, disable_progbar=not self.verbose)  # Progbar if verbose
        self.mcmc.run(self.prior_params, logits, labels)

        self._update_posterior_params()
        self.timestep += 1

        return self.mcmc

    def _update_prior_params(self):
        """ Updates the prior parameters using the ADF posterior from the previous timestep, plus the drift.

        If this is the first batch, i.e. timestep == 0, do nothing.
        """
        if self.timestep > 0:
            self.prior_params['mu_beta'] = self.posterior_params['mu_beta']
            self.prior_params['sigma_beta'] = self.posterior_params['sigma_beta'] + self.sigma_drift

    def _update_posterior_params(self):
        """ Fits a normal distribution to the current beta samples using moment matching.
        """
        beta_samples = self.get_current_posterior_samples()
        self.posterior_params['mu_beta'] = beta_samples.mean().item()
        self.posterior_params['sigma_beta'] = beta_samples.std().item()

    def get_current_posterior_samples(self):
        """ Returns the current posterior samples for beta.
        """
        if self.mcmc is None:
            return None

        posterior_samples = self.mcmc.get_samples()['beta']

        return posterior_samples

    def calibrate(self, logit):
        """ Calibrates the given batch of logits using the current posterior samples.

        Args:
            logit: tensor ; shape (batch_size, num_classes)
        """
        # Get beta samples
        beta_samples = self.get_current_posterior_samples()  # Shape (num_samples, num_classes)

        # Map betas to temperatures
        temperature_samples = torch.exp(-1. * beta_samples)  # Shape (num_samples, )

        # Get a batch of logits for each sampled temperature
        # Shape (num_samples, batch_size, num_classes)
        tempered_logit_samples = temperature_samples.view(-1, 1, 1) * logit

        # Softmax the sampled logits to get sampled probabilities
        prob_samples = softmax(tempered_logit_samples, dim=2)  # Shape (num_samples, batch_size, num_classes)

        # Average over the sampled probabilities to get Monte Carlo estimate
        calibrated_probs = prob_samples.mean(dim=0)  # Shape (batch_size, num_classes)

        return calibrated_probs

    def get_MAP_temperature(self, logits, labels):
        """ Performs MAP estimation using the current prior and given data.
         NB: This should only be called after .update() if used in a sequential setting, as this method
         does not update the prior with sigma_drift.

         Further note: This method is not very robust -- make sure to check for convergence if used.

         See: https://pyro.ai/examples/mle_map.html
         """
        pyro.clear_param_store()
        svi = pyro.infer.SVI(model=bt_model, guide=MAP_guide,
                             optim=pyro.optim.Adam({'lr': 0.001}), loss=pyro.infer.Trace_ELBO())

        loss = []
        num_steps = 5000
        for _ in range(num_steps):
            loss.append(svi.step(self.prior_params, logits, labels))

        eps = 2e-2
        loss_sddev = np.std(loss[-25:])
        if loss_sddev > eps:
            warnings.warn('MAP optimization may not have converged ; sddev {}'.format(loss_sddev))
            print('Here is the last few loss terms for inspection: \n', loss[-50:])

        MAP_temperature = torch.exp(pyro.param('beta_MAP')).item()
        return MAP_temperature


# NB: Labels must be in [0, 1, 2, . . . num_classes - 1] !
def bt_model(prior_params, logits, labels):
    """ This function defines the Pyro model for BT.
    """
    n_obs = logits.shape[0]  # Batch size

    # Prior over global temperature Beta ~ N( beta_mu, beta_sigma^2 )
    prior_beta_mu = prior_params['mu_beta']
    prior_beta_sigma = prior_params['sigma_beta']
    beta = pyro.sample('beta', dist.Normal(prior_beta_mu, prior_beta_sigma))  # Shape (1, )

    probs = softmax(torch.exp(-1. * beta) * logits, dim=1)  # Shape (n_obs, n_classes) ; tempered probabilities

    # Observation plate ; vectorized
    with pyro.plate('obs', size=n_obs):
        a = pyro.sample('cat_obs', dist.Categorical(probs=probs), obs=labels)


def MAP_guide(prior_params, logits, labels):
    """ Defines a guide for use in MAP inference. """
    beta_MAP = pyro.param('beta_MAP', torch.tensor(1., requires_grad=True))
    pyro.sample('beta', dist.Delta(beta_MAP))
