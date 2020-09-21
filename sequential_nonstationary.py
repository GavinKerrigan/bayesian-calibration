import pathlib
import yaml
import numpy as np
import time
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import Subset, DataLoader
from torch.nn.functional import softmax
from torch.nn import NLLLoss

from utils import utils, model_utils, data_utils
from utils.metrics import *
from calibration_methods.bayesian_tempering import BayesianTemperingCalibrator as BT

from calibration_methods.sequential_temperature_scaling import MovingWindowTS, SequentialSGDTS

""" This file runs a sequential non-stationary experiment from a given config .yml file. 
"""


def run_experiment_rotate(model, calibration_dataset, eval_dataset, angle_schedule, **kwargs):
    # Initializing some various useful things
    t0 = time.time()
    nll = NLLLoss()
    pd_cols = 1 + np.arange(kwargs['num_timesteps'])

    acc = []

    # Shuffle to GPU ; necessary for this experiment as we need to forward-pass frequently
    assert torch.cuda.is_available(), 'You will want GPU for this experiment :~)'
    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    model = model.to(gpu)
    # calibration_dataset = calibration_dataset.to(gpu)
    # eval_dataset = eval_dataset.to(gpu)

    # TODO: Better way to store metrics.
    # Storing our metrics
    # Metrics used: ECE, NLL
    # Methods: MW-TS, SGD-TS, B-TS, B-TS-MAP
    ece_nocal = []
    ece_mwts = {window_size: [] for window_size in kwargs['window_sizes']}
    ece_tsgd = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
    ece_bt = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}
    ece_bt_MAP = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}

    nll_nocal = []
    nll_mwts = {window_size: [] for window_size in kwargs['window_sizes']}
    nll_tsgd = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
    nll_bt = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}
    nll_bt_MAP = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}

    # Progress bars
    for run in tqdm(range(kwargs['num_runs']), desc='Run'):
        # Set up metrics for this run
        ece_nocal_run = []
        ece_mwts_run = {window_size: [] for window_size in kwargs['window_sizes']}
        ece_tsgd_run = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
        ece_bt_run = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}
        ece_bt_MAP_run = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}

        nll_nocal_run = []
        nll_mwts_run = {window_size: [] for window_size in kwargs['window_sizes']}
        nll_tsgd_run = {num_grad_steps: [] for num_grad_steps in kwargs['sgd_steps']}
        nll_bt_run = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}
        nll_bt_MAP_run = {sigma_drift: [] for sigma_drift in kwargs['sigma_drift']}

        # Define our calibrators
        mwts_models = {window_size: MovingWindowTS(window_size=window_size) for window_size in kwargs['window_sizes']}
        tsgd_models = {num_grad_steps: SequentialSGDTS(num_grad_steps=num_grad_steps)
                       for num_grad_steps in kwargs['sgd_steps']}
        bt_models = {sigma_drift: BT(kwargs['prior_params'], kwargs['num_classes'],
                                     sigma_drift=sigma_drift, **kwargs['mcmc_params'], verbose=False)
                     for sigma_drift in kwargs['sigma_drift']}

        for i in tqdm(range(kwargs['num_timesteps']), desc='Timestep', leave=False):
            # =============================================
            # Get eval/cal logits&labels, after rotation by angle_schedule[i]
            # =============================================
            with torch.no_grad():
                # Perturb eval_set and forward pass model
                eval_dataset.dataset.set_angle(angle_schedule[i])
                eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=0)
                eval_logits, eval_labels = model_utils.forward_pass(model, eval_loader, kwargs['num_classes'],
                                                                    device=gpu, verbose=False)
                # I think this is needed:
                eval_logits = eval_logits.to(cpu)
                eval_labels = eval_labels.to(cpu)

                # Get accuracy for this timestep; no need to do on every run as the eval set is constant
                if run == 0:
                    acc.append((eval_labels == eval_logits.argmax(dim=1)).double().mean().item())

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
                                                                  kwargs['num_classes'], device=gpu, verbose=False)
                cal_logits = cal_logits.to(cpu)
                cal_labels = cal_labels.to(cpu)

            # =============================================
            # Perform calibration with the various methods
            # =============================================

            # ---- | No calibration
            nocal_eval_probs = softmax(eval_logits, dim=1)
            ece_nocal_run.append(expected_calibration_error(nocal_eval_probs, eval_labels))
            nll_nocal_run.append(nll(nocal_eval_probs.log(), eval_labels.long()).item())

            # ----| MWTS
            for window_size in kwargs['window_sizes']:
                mwts_models[window_size].update(cal_logits, cal_labels.long())
                eval_probs_mwts = mwts_models[window_size].calibrate(eval_logits)
                # --------| Get metrics
                ece_mwts_run[window_size].append(expected_calibration_error(eval_probs_mwts, eval_labels))
                nll_mwts_run[window_size].append(nll(eval_probs_mwts.log(), eval_labels.long()).item())

            # ----| TSGD
            for num_grad_steps in kwargs['sgd_steps']:
                tsgd_models[num_grad_steps].update(cal_logits, cal_labels.long())
                eval_probs_tsgd = tsgd_models[num_grad_steps].calibrate(eval_logits)
                # --------| Get metrics
                ece_tsgd_run[num_grad_steps].append(expected_calibration_error(eval_probs_tsgd, eval_labels))
                nll_tsgd_run[num_grad_steps].append(nll(eval_probs_tsgd.log(), eval_labels.long()).item())

            # ----| BT
            # TODO: MAP
            for sigma_drift in kwargs['sigma_drift']:
                bt_models[sigma_drift].update(cal_logits, cal_labels)
                eval_probs_bt = bt_models[sigma_drift].calibrate(eval_logits)
                # --------| Get metrics
                ece_bt_run[sigma_drift].append(expected_calibration_error(eval_probs_bt, eval_labels))
                nll_bt_run[sigma_drift].append(nll(eval_probs_bt.log(), eval_labels.long()).item())
                # ---- | Run MAP estimation for comparison
                MAP_temperature = bt_models[sigma_drift].get_MAP_temperature(cal_logits, cal_labels)
                MAP_eval_probs = softmax(1./MAP_temperature * eval_logits, dim=1)
                ece_bt_MAP_run[sigma_drift].append(expected_calibration_error(MAP_eval_probs, eval_labels))
                nll_bt_MAP_run[sigma_drift].append(nll(MAP_eval_probs.log(), eval_labels.long()).item())

        # Store run data
        ece_nocal.append(ece_nocal_run)
        nll_nocal.append(nll_nocal_run)
        for window_size in kwargs['window_sizes']:
            ece_mwts[window_size].append(ece_mwts_run[window_size])
            nll_mwts[window_size].append(ece_mwts_run[window_size])
        for num_grad_steps in kwargs['sgd_steps']:
            ece_tsgd[num_grad_steps].append(ece_tsgd_run[num_grad_steps])
            nll_tsgd[num_grad_steps].append(nll_tsgd_run[num_grad_steps])
        for sigma_drift in kwargs['sigma_drift']:
            ece_bt[sigma_drift].append(ece_bt_run[sigma_drift])
            nll_bt[sigma_drift].append(nll_bt_run[sigma_drift])
            ece_bt_MAP[sigma_drift].append(ece_bt_MAP_run[sigma_drift])
            nll_bt_MAP[sigma_drift].append(nll_bt_MAP_run[sigma_drift])

        # Save after every run in case the run crashes
        pd.DataFrame(data=ece_nocal, columns=pd_cols).to_csv('ece_nocal.csv')
        pd.DataFrame(data=nll_nocal, columns=pd_cols).to_csv('nll_nocal.csv')
        for window_size in kwargs['window_sizes']:
            pd.DataFrame(data=ece_mwts[window_size], columns=pd_cols).to_csv('ece_mwts{}.csv'.format(window_size))
            pd.DataFrame(data=nll_mwts[window_size], columns=pd_cols).to_csv('nll_mwts{}.csv'.format(window_size))
        for num_grad_steps in kwargs['sgd_steps']:
            pd.DataFrame(data=ece_tsgd[num_grad_steps],
                         columns=pd_cols).to_csv('ece_tsgd{}.csv'.format(num_grad_steps))
            pd.DataFrame(data=nll_tsgd[num_grad_steps],
                         columns=pd_cols).to_csv('nll_tsgd{}.csv'.format(num_grad_steps))
        for sigma_drift in kwargs['sigma_drift']:
            pd.DataFrame(data=ece_bt[sigma_drift], columns=pd_cols).to_csv('ece_bt{}.csv'.format(sigma_drift))
            pd.DataFrame(data=nll_bt[sigma_drift], columns=pd_cols).to_csv('nll_bt{}.csv'.format(sigma_drift))
            pd.DataFrame(data=ece_bt_MAP[sigma_drift], columns=pd_cols).to_csv('ece_bt{}_MAP.csv'.format(sigma_drift))
            pd.DataFrame(data=nll_bt_MAP[sigma_drift], columns=pd_cols).to_csv('nll_bt{}_MAP.csv'.format(sigma_drift))

    # Save after experiment terminates
    for window_size in kwargs['window_sizes']:
        pd.DataFrame(data=ece_mwts[window_size], columns=pd_cols).to_csv('ece_mwts{}.csv'.format(window_size))
        pd.DataFrame(data=nll_mwts[window_size], columns=pd_cols).to_csv('nll_mwts{}.csv'.format(window_size))
    for num_grad_steps in kwargs['sgd_steps']:
        pd.DataFrame(data=ece_tsgd[num_grad_steps],
                     columns=pd_cols).to_csv('ece_tsgd{}.csv'.format(num_grad_steps))
        pd.DataFrame(data=nll_tsgd[num_grad_steps],
                     columns=pd_cols).to_csv('nll_tsgd{}.csv'.format(num_grad_steps))
    for sigma_drift in kwargs['sigma_drift']:
        pd.DataFrame(data=ece_bt[sigma_drift], columns=pd_cols).to_csv('ece_bt{}.csv'.format(sigma_drift))
        pd.DataFrame(data=nll_bt[sigma_drift], columns=pd_cols).to_csv('nll_bt{}.csv'.format(sigma_drift))
        pd.DataFrame(data=ece_bt_MAP[sigma_drift], columns=pd_cols).to_csv('ece_bt{}_MAP.csv'.format(sigma_drift))
        pd.DataFrame(data=nll_bt_MAP[sigma_drift], columns=pd_cols).to_csv('nll_bt{}_MAP.csv'.format(sigma_drift))

    t1 = time.time()
    pd.DataFrame(acc).to_csv('accuracy.csv')
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

    angle_schedule_type = config.pop('angle_schedule')
    if angle_schedule_type == 'linear':
        angle_schedule = linear_angle_schedule(config['angle_rate'], config['num_timesteps'])
    elif angle_schedule_type == 'triangular':
        angle_schedule = linear_angle_schedule(config['angle_rate'], config['num_timesteps'])

    return run_experiment_rotate(model, calibration_dataset, eval_dataset, angle_schedule, **config)


def linear_angle_schedule(rate, num_timesteps):
    # Defines an angle schedule that increases the angle by rate at each timestep.
    angle_schedule = rate * np.arange(num_timesteps, dtype=float)
    return angle_schedule


def triangular_angle_schedule(rate, num_timesteps):
    # Defines an angle schedule that increases the angle by rate until num_timesteps/2, and then decreases
    angle_schedule = [rate * i if i <= num_timesteps / 2.
                      else rate * (num_timesteps - i)
                      for i in range(num_timesteps)]
    return angle_schedule


if __name__ == '__main__':
    config_file = 'experiments/config/sequential_stationary.yml'
    print('config file: {}'.format(config_file))
    run_from_config(config_file)
