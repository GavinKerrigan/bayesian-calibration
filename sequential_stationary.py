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

from calibration_methods.sequential_temperature_scaling import SequentialBatchTS, SequentialSGDTS

""" This file runs a non-sequential experiment from a given config .yml file. 
"""


def run_experiment(model, calibration_dataset, eval_dataset, **kwargs):
    t0 = time.time()
    nll = NLLLoss()
    pd_cols = 1 + np.arange(kwargs['num_timesteps'])
    # Get model logits / labels on evaluation set
    with torch.no_grad():
        eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=0)
        eval_logits, eval_labels = model_utils.forward_pass(model, eval_loader, kwargs['num_classes'])

    # Storing our metrics -- probably a better way to do this but whatever
    # Methods: TSB, TSGD, BT
    cw_ece_tsb = []
    cw_ece_tsgd = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
    cw_ece_bt = []

    ece_tsb = []
    ece_tsgd = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
    ece_bt = []

    nll_tsb = []
    nll_tsgd = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
    nll_bt = []

    for run in range(kwargs['num_runs']):
        print('=' * 15)
        print('Run {} of {}'.format(run + 1, kwargs['num_runs']))
        print('=' * 15)

        cw_ece_tsb_run = []
        cw_ece_tsgd_run = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
        cw_ece_bt_run = []

        ece_tsb_run = []
        ece_tsgd_run = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
        ece_bt_run = []

        nll_tsb_run = []
        nll_tsgd_run = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
        nll_bt_run = []

        # Get a shuffled calibration loader for the run
        calibration_batch_loader = DataLoader(calibration_dataset, batch_size=kwargs['batch_size'],
                                              shuffle=True, num_workers=0)
        # Forward pass batch through model ; get all logits and labels
        cal_logits, cal_labels = model_utils.forward_pass(model, calibration_batch_loader, kwargs['num_classes'])

        # Define our calibrators
        tsb_model = SequentialBatchTS()
        tsgd_models = {num_grad_steps : SequentialSGDTS(num_grad_steps=num_grad_steps)
                      for num_grad_steps in kwargs['sgd_steps']}
        bt_model = BT(kwargs['prior_params'], kwargs['num_classes'], **kwargs['mcmc_params'])

        for i in range(kwargs['num_timesteps']):
            print('-' * 15)
            print('Running timestep: {} of {}'.format(i, kwargs['num_timesteps']))
            print('-' * 15)

            # =============================================
            # Perform calibration with the various methods
            # =============================================

            # left/right indexes of batch for convenience
            l = i * kwargs['batch_size']
            r = (i + 1) * kwargs['batch_size']

            # ----| TSB
            tsb_model.update(cal_logits[l:r], cal_labels[l:r].long())
            eval_probs_tsb = tsb_model.calibrate(eval_logits)
            # --------| Get metrics
            cw_ece_tsb_run.append(classwise_ece(eval_probs_tsb, eval_labels.int().numpy()))
            ece_tsb_run.append(expected_calibration_error(eval_probs_tsb, eval_labels))
            nll_tsb_run.append(nll(eval_probs_tsb.log(), eval_labels.long()).item())

            # ----| TSGD
            for num_grad_steps in kwargs['sgd_steps']:
                tsgd_models[num_grad_steps].update(cal_logits[l:r], cal_labels[l:r].long())
                eval_probs_tsgd = tsgd_models[num_grad_steps].calibrate(eval_logits)
                # --------| Get metrics
                cw_ece_tsgd_run[num_grad_steps].append(classwise_ece(eval_probs_tsgd, eval_labels.int().numpy()))
                ece_tsgd_run[num_grad_steps].append(expected_calibration_error(eval_probs_tsgd, eval_labels))
                nll_tsgd_run[num_grad_steps].append(nll(eval_probs_tsgd.log(), eval_labels.long()).item())

            # ----| BT
            bt_model.update(cal_logits[l:r], cal_labels[l:r])
            eval_probs_bt = bt_model.calibrate(eval_logits)
            # --------| Get metrics
            cw_ece_bt_run.append(classwise_ece(eval_probs_bt, eval_labels.int().numpy()))
            ece_bt_run.append(expected_calibration_error(eval_probs_bt, eval_labels))
            nll_bt_run.append(nll(eval_probs_bt.log(), eval_labels.long()).item())

        cw_ece_tsb.append(cw_ece_tsb_run)
        ece_tsb.append(ece_tsb_run)
        nll_tsb.append(nll_tsb_run)

        for num_grad_steps in kwargs['sgd_steps']:
            cw_ece_tsgd[num_grad_steps].append(cw_ece_tsgd_run[num_grad_steps])
            ece_tsgd[num_grad_steps].append(ece_tsgd_run[num_grad_steps])
            nll_tsgd[num_grad_steps].append(nll_tsgd_run[num_grad_steps])

        cw_ece_bt.append(cw_ece_bt_run)
        ece_bt.append(ece_bt_run)
        nll_bt.append(nll_bt_run)

        # Save after every run in case I need to end early / the run crashes
        pd.DataFrame(data=cw_ece_tsb, columns=pd_cols).to_csv('cw_ece_tsb.csv')
        pd.DataFrame(data=ece_tsb, columns=pd_cols).to_csv('ece_tsb.csv')
        pd.DataFrame(data=nll_tsb, columns=pd_cols).to_csv('nll_tsb.csv')

        for num_grad_steps in kwargs['sgd_steps']:
            pd.DataFrame(data=cw_ece_tsgd[num_grad_steps],
                         columns=pd_cols).to_csv('cw_ece_tsgd{}.csv'.format(num_grad_steps))
            pd.DataFrame(data=ece_tsgd[num_grad_steps],
                         columns=pd_cols).to_csv('ece_tsgd{}.csv'.format(num_grad_steps))
            pd.DataFrame(data=nll_tsgd[num_grad_steps],
                         columns=pd_cols).to_csv('nll_tsgd{}.csv'.format(num_grad_steps))

        pd.DataFrame(data=cw_ece_bt, columns=pd_cols).to_csv('cw_ece_bt.csv')
        pd.DataFrame(data=ece_bt, columns=pd_cols).to_csv('ece_bt.csv')
        pd.DataFrame(data=nll_bt, columns=pd_cols).to_csv('nll_bt.csv')

    pd.DataFrame(data=cw_ece_tsb, columns=pd_cols).to_csv('cw_ece_tsb.csv')
    pd.DataFrame(data=ece_tsb, columns=pd_cols).to_csv('ece_tsb.csv')
    pd.DataFrame(data=nll_tsb, columns=pd_cols).to_csv('nll_tsb.csv')

    for num_grad_steps in kwargs['sgd_steps']:
        pd.DataFrame(data=cw_ece_tsgd[num_grad_steps],
                     columns=pd_cols).to_csv('cw_ece_tsgd{}.csv'.format(num_grad_steps))
        pd.DataFrame(data=ece_tsgd[num_grad_steps],
                     columns=pd_cols).to_csv('ece_tsgd{}.csv'.format(num_grad_steps))
        pd.DataFrame(data=nll_tsgd[num_grad_steps],
                     columns=pd_cols).to_csv('nll_tsgd{}.csv'.format(num_grad_steps))

    pd.DataFrame(data=cw_ece_bt, columns=pd_cols).to_csv('cw_ece_bt.csv')
    pd.DataFrame(data=ece_bt, columns=pd_cols).to_csv('ece_bt.csv')
    pd.DataFrame(data=nll_bt, columns=pd_cols).to_csv('nll_bt.csv')

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
    config_file = 'experiments/config/sequential_stationary.yml'
    print('config file: {}'.format(config_file))
    run_from_config(config_file)
