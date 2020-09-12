import pathlib
import yaml
import numpy as np
import time
import pandas as pd
from torch.utils.data import Subset, DataLoader
from torch.nn.functional import softmax
from torch.nn import NLLLoss

from utils import utils, model_utils, data_utils
from utils.metrics import *
from calibration_methods.hierarchical_bayes import HierarchicalBayesianCalibrator as HBC
from calibration_methods.bayesian_tempering import BayesianTemperingCalibrator as BT
from calibration_methods.maximum_likelihood import vector_scaling, temperature_scaling

""" This file runs a non-sequential experiment from a given config .yml file. 
"""


def run_experiment(model, calibration_dataset, eval_dataset, **kwargs):
    t0 = time.time()
    nll = NLLLoss()
    # Get model logits / labels on evaluation set
    with torch.no_grad():
        eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=0)
        eval_logits, eval_labels = model_utils.forward_pass(model, eval_loader, kwargs['num_classes'])

    # Storing our metrics -- probably a better way to do this but whatever
    # Methods: TS, VS, HBC, BTS
    cw_ece_ts = []
    cw_ece_bt = []
    cw_ece_vs = []
    cw_ece_hbc = []

    ece_ts = []
    ece_bt = []
    ece_vs = []
    ece_hbc = []

    nll_ts = []
    nll_bt = []
    nll_vs = []
    nll_hbc = []
    for run in range(kwargs['num_runs']):
        print('=' * 15)
        print('Run {} of {}'.format(run + 1, kwargs['num_runs']))
        print('=' * 15)
        cw_ece_ts_run = []
        cw_ece_bt_run = []
        cw_ece_vs_run = []
        cw_ece_hbc_run = []

        ece_ts_run = []
        ece_bt_run = []
        ece_vs_run = []
        ece_hbc_run = []

        nll_ts_run = []
        nll_bt_run = []
        nll_vs_run = []
        nll_hbc_run = []
        for batch_size in kwargs['batch_size']:
            print('-'*15)
            print('Running batch size: {}'.format(batch_size))
            print('-' * 15)
            # Get a subset of the calibration dataset of size batch size
            subset_idxs = np.random.choice(len(calibration_dataset), size=batch_size, replace=False)
            batch = Subset(calibration_dataset, subset_idxs)
            calibration_batch_loader = DataLoader(batch, batch_size=256, shuffle=False, num_workers=0)

            # Forward pass batch through model ; get logits and labels
            logits, labels = model_utils.forward_pass(model, calibration_batch_loader, kwargs['num_classes'])

            # =============================================
            # Perform calibration with the various methods
            # =============================================

            # ----| Temperature scaling
            ts_temperature = temperature_scaling(logits, labels.long())['temperature']
            eval_probs_ts = softmax(1. / ts_temperature * eval_logits, dim=1)
            # --------| Get metrics
            cw_ece_ts_run.append(classwise_ece(eval_probs_ts, eval_labels.int().numpy()))
            ece_ts_run.append(expected_calibration_error(eval_probs_ts, eval_labels))
            nll_ts_run.append(nll(eval_probs_ts.log(), eval_labels.long()).item())

            # ----| Vector scaling
            vs_temperature = vector_scaling(logits, labels.long(), bias=False)['temperature']
            eval_probs_vs = softmax(1. / vs_temperature * eval_logits, dim=1)
            # --------| Get metrics
            cw_ece_vs_run.append(classwise_ece(eval_probs_vs, eval_labels.int().numpy()))
            ece_vs_run.append(expected_calibration_error(eval_probs_vs, eval_labels))
            nll_vs_run.append(nll(eval_probs_vs.log(), eval_labels.long()).item())

            # ----| HBC
            hbc_model = HBC(kwargs['prior_params_hbc'], kwargs['num_classes'], **kwargs['mcmc_params'],
                            delta_constraint='hard')
            hbc_model.update(logits, labels)
            eval_probs_hbc = hbc_model.calibrate(eval_logits)
            # --------| Get metrics
            cw_ece_hbc_run.append(classwise_ece(eval_probs_hbc, eval_labels.int().numpy()))
            ece_hbc_run.append(expected_calibration_error(eval_probs_hbc, eval_labels))
            nll_hbc_run.append(nll(eval_probs_hbc.log(), eval_labels.long()).item())

            # ----| HBC
            bt_model = BT(kwargs['prior_params_bt'], kwargs['num_classes'], **kwargs['mcmc_params'])
            bt_model.update(logits, labels)
            eval_probs_bt = bt_model.calibrate(eval_logits)
            # --------| Get metrics
            cw_ece_bt_run.append(classwise_ece(eval_probs_bt, eval_labels.int().numpy()))
            ece_bt_run.append(expected_calibration_error(eval_probs_bt, eval_labels))
            nll_bt_run.append(nll(eval_probs_bt.log(), eval_labels.long()).item())

        cw_ece_vs.append(cw_ece_vs_run)
        ece_vs.append(ece_vs_run)
        nll_vs.append(nll_vs_run)

        cw_ece_ts.append(cw_ece_ts_run)
        ece_ts.append(ece_ts_run)
        nll_ts.append(nll_ts_run)

        cw_ece_hbc.append(cw_ece_hbc_run)
        ece_hbc.append(ece_hbc_run)
        nll_hbc.append(nll_hbc_run)

        cw_ece_bt.append(cw_ece_bt_run)
        ece_bt.append(ece_bt_run)
        nll_bt.append(nll_bt_run)

        # Hopefully this works... trying to save after every run in case I need to end early.
        pd.DataFrame(data=cw_ece_vs, columns=kwargs['batch_size']).to_csv('cw_ece_vs.csv')
        pd.DataFrame(data=ece_vs, columns=kwargs['batch_size']).to_csv('ece_vs.csv')
        pd.DataFrame(data=nll_vs, columns=kwargs['batch_size']).to_csv('nll_vs.csv')

        pd.DataFrame(data=cw_ece_ts, columns=kwargs['batch_size']).to_csv('cw_ece_ts.csv')
        pd.DataFrame(data=ece_ts, columns=kwargs['batch_size']).to_csv('ece_ts.csv')
        pd.DataFrame(data=nll_ts, columns=kwargs['batch_size']).to_csv('nll_ts.csv')

        pd.DataFrame(data=cw_ece_hbc, columns=kwargs['batch_size']).to_csv('cw_ece_hbc.csv')
        pd.DataFrame(data=ece_hbc, columns=kwargs['batch_size']).to_csv('ece_hbc.csv')
        pd.DataFrame(data=nll_hbc, columns=kwargs['batch_size']).to_csv('nll_hbc.csv')

        pd.DataFrame(data=cw_ece_bt, columns=kwargs['batch_size']).to_csv('cw_ece_bt.csv')
        pd.DataFrame(data=ece_bt, columns=kwargs['batch_size']).to_csv('ece_bt.csv')
        pd.DataFrame(data=nll_bt, columns=kwargs['batch_size']).to_csv('nll_bt.csv')

    pd.DataFrame(data=cw_ece_vs, columns=kwargs['batch_size']).to_csv('cw_ece_vs.csv')
    pd.DataFrame(data=ece_vs, columns=kwargs['batch_size']).to_csv('ece_vs.csv')
    pd.DataFrame(data=nll_vs, columns=kwargs['batch_size']).to_csv('nll_vs.csv')

    pd.DataFrame(data=cw_ece_ts, columns=kwargs['batch_size']).to_csv('cw_ece_ts.csv')
    pd.DataFrame(data=ece_ts, columns=kwargs['batch_size']).to_csv('ece_ts.csv')
    pd.DataFrame(data=nll_ts, columns=kwargs['batch_size']).to_csv('nll_ts.csv')

    pd.DataFrame(data=cw_ece_hbc, columns=kwargs['batch_size']).to_csv('cw_ece_hbc.csv')
    pd.DataFrame(data=ece_hbc, columns=kwargs['batch_size']).to_csv('ece_hbc.csv')
    pd.DataFrame(data=nll_hbc, columns=kwargs['batch_size']).to_csv('nll_hbc.csv')

    pd.DataFrame(data=cw_ece_bt, columns=kwargs['batch_size']).to_csv('cw_ece_bt.csv')
    pd.DataFrame(data=ece_bt, columns=kwargs['batch_size']).to_csv('ece_bt.csv')
    pd.DataFrame(data=nll_bt, columns=kwargs['batch_size']).to_csv('nll_bt.csv')

    t1 = time.time()
    print('\n\n\nFinished: runtime (s): {:.2f}'.format(t1 - t0))


def run_from_config(config_fpath):
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

    return run_experiment(model, calibration_dataset, eval_dataset, **config)


if __name__ == '__main__':
    config_file = 'experiments/config/small_data.yml'
    print('config file: {}'.format(config_file))
    run_from_config(config_file)
