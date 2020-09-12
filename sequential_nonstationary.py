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

""" This file runs a sequential non-stationary experiment from a given config .yml file. 
"""

# TODO: Moving window batch-TS

def run_experiment_rotate(model, calibration_dataset, eval_dataset, angle_schedule, **kwargs):
    # Initializing some various useful things
    t0 = time.time()
    nll = NLLLoss()
    pd_cols = 1 + np.arange(kwargs['num_timesteps'])

    # Shuffle to GPU ; necessary for this experiment as we need to forward-pass frequently
    assert torch.cuda.is_available(), 'You will want GPU for this experiment :~)'
    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    model = model.to(gpu)
    calibration_dataset = calibration_dataset.to(gpu)
    eval_dataset = eval_dataset.to(gpu)

    # TODO: Better way to store metrics.
    # Storing our metrics
    # Metrics used: ECE, NLL
    # Methods: TSB, TSGD, BT
    ece_tsb = []
    ece_tsgd = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
    ece_bt = []

    nll_tsb = []
    nll_tsgd = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
    nll_bt = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}

    for run in range(kwargs['num_runs']):
        print('=' * 15)
        print('Run {} of {}'.format(run + 1, kwargs['num_runs']))
        print('=' * 15)

        # Set up metrics for this run
        ece_tsb_run = []
        ece_tsgd_run = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
        ece_bt_run = []

        nll_tsb_run = []
        nll_tsgd_run = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
        nll_bt_run = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}

        # Define our calibrators
        tsb_model = SequentialBatchTS()
        tsgd_models = {num_grad_steps: SequentialSGDTS(num_grad_steps=num_grad_steps)
                       for num_grad_steps in kwargs['sgd_steps']}
        bt_models = {sigma_drift: BT(kwargs['prior_params'], kwargs['num_classes'],
                                     sigma_drift=sigma_drift, **kwargs['mcmc_params'])
                     for sigma_drift in kwargs['sigma_drift']}

        for i in range(kwargs['num_timesteps']):
            print('-' * 15)
            print('Running timestep: {} of {}'.format(i, kwargs['num_timesteps']))
            print('-' * 15)

            # =============================================
            # Get eval/cal logits&labels, after rotation by angle_schedule[i]
            # =============================================
            with torch.no_grad():
                # Perturb eval_set and forward pass model
                eval_dataset.dataset.set_angle(angle_schedule[i])
                eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=0)
                eval_logits, eval_labels = model_utils.forward_pass(model, eval_loader, kwargs['num_classes'])
                # I think this is needed:
                eval_logits = eval_logits.to(cpu)
                eval_labels = eval_labels.to(cpu)

                # Perturb the calibration data
                # (NB: not really necessary as we perturb the eval set, which shares the .dataset field)
                calibration_dataset.dataset.set_angle(angle_schedule[i])
                # Get a random batch of calibration data of size batch_size
                calibration_batch_idxs = np.random.choice(len(calibration_dataset),
                                                          size=kwargs['batch_size'], replace=False)
                calibration_batch = Subset(calibration_dataset, calibration_batch_idxs)
                calibration_batch_loader = DataLoader(calibration_batch, batch_size=kwargs['batch_size'],
                                                      shuffle=False, num_workers=0)
                # Forward pass calibration batch through model
                cal_logits, cal_labels = model_utils.forward_pass(model, calibration_batch_loader,
                                                                  kwargs['num_classes'])
                cal_logits = cal_logits.to(cpu)
                cal_labels = cal_labels.to(cpu)

            # =============================================
            # Perform calibration with the various methods
            # =============================================

            # left/right indexes of batch for convenience

            # ----| TSB
            tsb_model.update(cal_logits, cal_labels.long())
            eval_probs_tsb = tsb_model.calibrate(eval_logits)
            # --------| Get metrics
            ece_tsb_run.append(expected_calibration_error(eval_probs_tsb, eval_labels))
            nll_tsb_run.append(nll(eval_probs_tsb.log(), eval_labels.long()).item())

            # ----| TSGD
            for num_grad_steps in kwargs['sgd_steps']:
                tsgd_models[num_grad_steps].update(cal_logits, cal_labels.long())
                eval_probs_tsgd = tsgd_models[num_grad_steps].calibrate(eval_logits)
                # --------| Get metrics
                ece_tsgd_run[num_grad_steps].append(expected_calibration_error(eval_probs_tsgd, eval_labels))
                nll_tsgd_run[num_grad_steps].append(nll(eval_probs_tsgd.log(), eval_labels.long()).item())

            # ----| BT
            for sigma_drift in kwargs['sigma_drift']:
                bt_models[sigma_drift].update(cal_logits, cal_labels)
                eval_probs_bt = bt_models[sigma_drift].calibrate(eval_logits)
                # --------| Get metrics
                ece_bt_run[sigma_drift].append(expected_calibration_error(eval_probs_bt, eval_labels))
                nll_bt_run[sigma_drift].append(nll(eval_probs_bt.log(), eval_labels.long()).item())

        ece_tsb.append(ece_tsb_run)
        nll_tsb.append(nll_tsb_run)

        for num_grad_steps in kwargs['sgd_steps']:
            ece_tsgd[num_grad_steps].append(ece_tsgd_run[num_grad_steps])
            nll_tsgd[num_grad_steps].append(nll_tsgd_run[num_grad_steps])

        ece_bt.append(ece_bt_run)
        nll_bt.append(nll_bt_run)

        # Save after every run in case I need to end early / the run crashes
        pd.DataFrame(data=ece_tsb, columns=pd_cols).to_csv('ece_tsb.csv')
        pd.DataFrame(data=nll_tsb, columns=pd_cols).to_csv('nll_tsb.csv')

        for num_grad_steps in kwargs['sgd_steps']:
            pd.DataFrame(data=ece_tsgd[num_grad_steps],
                         columns=pd_cols).to_csv('ece_tsgd{}.csv'.format(num_grad_steps))
            pd.DataFrame(data=nll_tsgd[num_grad_steps],
                         columns=pd_cols).to_csv('nll_tsgd{}.csv'.format(num_grad_steps))

        pd.DataFrame(data=ece_bt, columns=pd_cols).to_csv('ece_bt.csv')
        pd.DataFrame(data=nll_bt, columns=pd_cols).to_csv('nll_bt.csv')

    pd.DataFrame(data=ece_tsb, columns=pd_cols).to_csv('ece_tsb.csv')
    pd.DataFrame(data=nll_tsb, columns=pd_cols).to_csv('nll_tsb.csv')

    for num_grad_steps in kwargs['sgd_steps']:
        pd.DataFrame(data=ece_tsgd[num_grad_steps],
                     columns=pd_cols).to_csv('ece_tsgd{}.csv'.format(num_grad_steps))
        pd.DataFrame(data=nll_tsgd[num_grad_steps],
                     columns=pd_cols).to_csv('nll_tsgd{}.csv'.format(num_grad_steps))

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
