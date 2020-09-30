import warnings
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

import numpy as np
import torch
from torch.nn.functional import softmax

""" This class implements the Bayesian vector scaling calibrator.
This is the 'standard' version with no fancy modelling. 
"""


class BayesianVSCalibrator:
    """ This class implements the Bayesian VS calibrator, with bias.
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

        # Prior parameters on beta / delta ; assumes each weight/bias is i.i.d from its respective distribution.
        self.prior_params = {'mu_beta': torch.empty(self.num_classes).fill_(prior_params['mu_beta']),
                             'sigma_beta': torch.empty(self.num_classes).fill_(prior_params['sigma_beta']),
                             'mu_delta': torch.empty(self.num_classes).fill_(prior_params['mu_delta']),
                             'sigma_delta': torch.empty(self.num_classes).fill_(prior_params['sigma_delta'])}

        # Posterior parameters after ADF
        # TODO
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
            print('\nInitializing VS model:\n'
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

        # TODO
        # self._update_prior_params()
        if self.verbose:
            print('--------| Updated priors: {}'.format(self.prior_params))
            print('--------| Running inference ')
        nuts_kernel = NUTS(bvs_model, **self.NUTS_params)
        self.mcmc = MCMC(nuts_kernel, **self.mcmc_params, disable_progbar=not self.verbose)  # Progbar if verbose
        self.mcmc.run(self.prior_params, logits, labels)

        # TODO
        # self._update_posterior_params()
        self.timestep += 1

        return self.mcmc

    def _update_prior_params(self):
        """ Updates the prior parameters using the ADF posterior from the previous timestep, plus the drift.

        If this is the first batch, i.e. timestep == 0, do nothing.
        """
        # TODO
        if self.timestep > 0:
            self.prior_params['mu_beta'] = self.posterior_params['mu_beta']
            self.prior_params['sigma_beta'] = self.posterior_params['sigma_beta'] + self.sigma_drift

    def _update_posterior_params(self):
        """ Fits a normal distribution to the current beta samples using moment matching.
        """
        # TODO
        beta_samples = self.get_current_posterior_samples()
        self.posterior_params['mu_beta'] = beta_samples.mean().item()
        self.posterior_params['sigma_beta'] = beta_samples.std().item()

    def get_current_posterior_samples(self):
        """ Returns the current posterior samples for beta.
        """
        if self.mcmc is None:
            return None

        return self.mcmc.get_samples()

    def calibrate(self, logit):
        """ Calibrates the given batch of logits using the current posterior samples.

        Args:
            logit: tensor ; shape (batch_size, num_classes)
        """
        # Get beta samples
        beta_samples = self.get_current_posterior_samples()['beta']  # Shape (num_samples, num_classes)
        delta_samples = self.get_current_posterior_samples()['delta']  # Shape (num_samples, num_classes)

        # Get a batch of logits for each sampled parameter vector
        # Shape (num_samples, batch_size, num_classes)
        tempered_logit_samples = beta_samples.view(-1, 1, self.num_classes) * logit + \
                                 delta_samples.view(-1, 1, self.num_classes)

        # Softmax the sampled logits to get sampled probabilities
        prob_samples = softmax(tempered_logit_samples, dim=2)  # Shape (num_samples, batch_size, num_classes)

        # Average over the sampled probabilities to get Monte Carlo estimate
        calibrated_probs = prob_samples.mean(dim=0)  # Shape (batch_size, num_classes)

        return calibrated_probs

    def get_MAP_temperature(self, logits, labels):
        """ Performs MAP estimation using the current prior and given data.
         NB: This should only be called after .update() if used in a sequential setting, as this method
         does not update the prior with sigma_drift.

         See: https://pyro.ai/examples/mle_map.html
         """
        pyro.clear_param_store()
        svi = pyro.infer.SVI(model=bvs_model, guide=MAP_guide,
                             optim=pyro.optim.Adam({'lr': 0.001}), loss=pyro.infer.Trace_ELBO())

        loss = []
        num_steps = 5000
        for _ in range(num_steps):
            loss.append(svi.step(self.prior_params, logits, labels))

        eps = 2e-2
        loss_sddev = np.std(loss[-25:])
        if loss_sddev > eps:
            warnings.warn('MAP optimization may not have converged ; sddev {}'.format(loss_sddev))

        beta_MAP = pyro.param('beta_MAP').detach()
        delta_MAP = pyro.param('delta_MAP').detach()
        return beta_MAP, delta_MAP


# NB: Labels must be in [0, 1, 2, . . . num_classes - 1] !
def bvs_model(prior_params, logits, labels):
    """ This function defines the Pyro model for BVS.
    """
    n_obs = logits.shape[0]  # Batch size
    n_cls = logits.shape[1]  # Number of classes

    # TODO: Question: Should I even bother constraining the weights to be positive here?
    # Current implementation is unconstrained.

    # Prior over temperatures Beta ~ N( beta_mu, beta_sigma^2 )
    prior_beta_mu = prior_params['mu_beta']  # Shape (n_classes, )
    prior_beta_sigma = prior_params['sigma_beta']  # Shape (n_classes, )
    beta = pyro.sample('beta', dist.Normal(prior_beta_mu, prior_beta_sigma))  # Shape (n_classes, )
    """
    beta = pyro.sample('beta', dist.MultivariateNormal(prior_beta_mu * torch.ones(n_cls),
                                                       covariance_matrix=prior_beta_sigma * torch.eye(n_cls)))
    """

    # Prior over delta's ; vectorized
    prior_delta_mu = prior_params['mu_delta']  # Shape (n_classes, )
    prior_delta_sigma = prior_params['sigma_delta']  # Shape (n_classes, )
    delta = pyro.sample('delta', dist.Normal(prior_delta_mu, prior_delta_sigma))  # Shape (n_classes, )
    """
    delta = pyro.sample('delta', dist.MultivariateNormal(prior_delta_mu * torch.ones(n_cls),
                                                        covariance_matrix=prior_delta_sigma * torch.eye(n_cls)))
    """
    tempered_logits = beta * logits + delta
    probs = softmax(tempered_logits, dim=1)  # Shape (n_obs, n_classes) ; tempered probabilities

    # Observation plate ; vectorized
    with pyro.plate('obs', size=n_obs):
        a = pyro.sample('cat_obs', dist.Categorical(probs=probs), obs=labels)


def MAP_guide(prior_params, logits, labels):
    """ Defines a guide for use in MAP inference. """
    n_cls = logits.shape[1]  # Num classes

    beta_MAP = pyro.param('beta_MAP', torch.ones(n_cls, requires_grad=True))
    delta_MAP = pyro.param('delta_MAP', torch.zeros(n_cls, requires_grad=True))
    pyro.sample('beta', dist.Delta(beta_MAP))
    pyro.sample('delta', dist.Delta(delta_MAP))
