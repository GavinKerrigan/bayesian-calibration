import pathlib
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import Subset, DataLoader
from torch.nn.functional import softmax
from torch.nn import NLLLoss

from utils import utils, model_utils, data_utils
from utils.metrics import *
from calibration_methods.hierarchical_bayes import HierarchicalBayesianCalibrator as HBC
from calibration_methods.maximum_likelihood import vector_scaling
from calibration import get_calibration_error as marginal_ce

""" This file runs a non-sequential experiment from a given config .yml file. 
"""


def run_experiment(model, calibration_dataset, eval_dataset, **kwargs):
    debias = False
    nll = NLLLoss()
    # Get model logits / labels on evaluation set
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=0)
    eval_logits, eval_labels = model_utils.forward_pass(model, eval_loader, kwargs['num_classes'])

    # Storing our metrics -- probably a better way to do this but whatever
    marginal_ce_vs = marginal_ce_hbc = []
    ece_vs = ece_hbc = []
    nll_vs = nll_hbc = []
    brier_vs = brier_hbc = []
    for run in range(kwargs['num_runs']):
        marginal_ce_vs_run = marginal_ce_hbc_run = []
        ece_vs_run = ece_hbc_run = []
        nll_vs_run = nll_hbc_run = []
        brier_vs_run = brier_hbc_run = []
        for batch_size in kwargs['batch_size']:
            # Get a subset of the calibration dataset of size batch size
            subset_idxs = np.random.choice(len(calibration_dataset), size=batch_size, replace=False)
            batch = Subset(calibration_dataset, subset_idxs)
            calibration_batch_loader = DataLoader(batch, batch_size=256, shuffle=False, num_workers=0)

            # Forward pass batch through model ; get logits and labels
            logits, labels = model_utils.forward_pass(model, calibration_batch_loader, kwargs['num_classes'])

            # Perform calibration with the various methods
            # ----| Vector scaling
            vs_temperature = vector_scaling(logits, labels.long())['temperature']
            eval_probs_vs = softmax(1. / vs_temperature * eval_logits, dim=1)
            # --------| Get metrics
            marginal_ce_vs_run.append(marginal_ce(eval_probs_vs, eval_labels.int().numpy(), debias=debias))
            ece_vs_run.append(expected_calibration_error(eval_probs_vs, eval_labels))
            nll_vs_run.append(nll(eval_probs_vs.log(), eval_labels.long()))
            brier_vs_run.append(brier_score(eval_probs_vs, eval_labels))

            # ----| HBC
            hbc_model = HBC(kwargs['prior_params'], kwargs['num_classes'], **kwargs['mcmc_params'])
            hbc_model.update(logits, labels)
            eval_probs_hbc = hbc_model.calibrate(eval_logits)
            # --------| Get metrics
            marginal_ce_hbc_run.append(marginal_ce(eval_probs_hbc, eval_labels.int().numpy(), debias=debias))
            ece_hbc_run.append(expected_calibration_error(eval_probs_hbc, eval_labels))
            nll_hbc_run.append(nll(eval_probs_hbc.log(), eval_labels.long()))
            brier_hbc_run.append(brier_score(eval_probs_hbc, eval_labels))

        marginal_ce_vs.append(marginal_ce_vs_run)
        ece_vs.append(ece_vs_run)
        nll_vs.append(nll_vs_run)
        brier_vs.append(brier_vs_run)

        marginal_ce_hbc.append(marginal_ce_hbc_run)
        ece_hbc.append(ece_hbc_run)
        nll_hbc.append(nll_hbc_run)
        brier_hbc.append(brier_hbc_run)

    pd.DataFrame(marginal_ce_vs, columns=range(kwargs['batch_size'])).to_pickle('marginal_ce_vs.pkl')
    pd.DataFrame(ece_vs, columns=range(kwargs['batch_size'])).to_pickle('ece_vs.pkl')
    pd.DataFrame(nll_vs, columns=range(kwargs['batch_size'])).to_pickle('nll_vs.pkl')
    pd.DataFrame(brier_vs, columns=range(kwargs['batch_size'])).to_pickle('brier_vs.pkl')

    pd.DataFrame(marginal_ce_hbc, columns=range(kwargs['batch_size'])).to_pickle('marginal_ce_hbc.pkl')
    pd.DataFrame(ece_hbc, columns=range(kwargs['batch_size'])).to_pickle('ece_hbc.pkl')
    pd.DataFrame(nll_hbc, columns=range(kwargs['batch_size'])).to_pickle('nll_hbc.pkl')
    pd.DataFrame(brier_hbc, columns=range(kwargs['batch_size'])).to_pickle('brier_hbc.pkl')


def main(config_fpath):
    # TODO: Set config file

    with open(config_fpath, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Set RNG seed
    if 'seed' in config:
        utils.set_seed(config['seed'])

    # Load our pre-trained model
    model = model_utils.load_trained_model(config.pop('model'), config['train_set'])
    # Get a fixed calibration / evaluation set
    calibration_dataset, eval_dataset = data_utils.get_cal_eval_split(config['test_set'], config['num_eval'])

    run_experiment(model, calibration_dataset, eval_dataset, **config)


if __name__ == '__main__':
    config_file = 'experiments/config/nonseq_test.yml'
    main(config_file)
