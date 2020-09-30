# Standard libs
import yaml
import numpy as np
import time
import pandas as pd
from tqdm.notebook import tqdm
# Torch utils
from torch.utils.data import Subset, DataLoader
from torch.nn.functional import softmax
from torch.nn import NLLLoss
# Custom utils
from utils import utils, model_utils, data_utils
from utils.metrics import *
# Calibration methods
from calibration_methods.bayesian_tempering import BayesianTemperingCalibrator as BT
from calibration_methods.maximum_likelihood import vector_scaling, temperature_scaling
from pycalib.calibration_methods import GPCalibration


""" This file runs a non-sequential experiment from a given config .yml file. 
"""


def run_experiment(model, calibration_dataset, eval_dataset, **kwargs):
    t0 = time.time()
    nll = NLLLoss()

    assert torch.cuda.is_available(), 'Requires GPU support'
    gpu = torch.device('cuda')
    cpu = torch.device('cpu')
    model = model.to(gpu)

    # Get model logits / labels on evaluation set
    with torch.no_grad():
        eval_loader = DataLoader(eval_dataset, batch_size=512, shuffle=False, num_workers=4)
        eval_logits, eval_labels = model_utils.forward_pass(model, eval_loader, kwargs['num_classes'],
                                                            device=gpu, verbose=False)
    eval_logits = eval_logits.to(cpu)
    eval_labels = eval_labels.to(cpu)

    # Evaluate ECE / NLL with no calibration
    ece_nocal = expected_calibration_error(softmax(eval_logits, dim=1), eval_labels)
    nll_nocal = nll(eval_logits, eval_labels.long())
    with open('metrics_nocal.txt', 'w') as f:
        f.write('ECE no calibration: {:.4f}'.format(ece_nocal))
        f.write('NLL no calibration: {:.4f}'.format(nll_nocal))

    # Storing our metrics -- probably a better way to do this but whatever
    # Methods: TS, VS, BTS, BTS-MAP, GPC
    ece_ts = []
    ece_bt = []
    ece_bt_MAP = []
    ece_vs = []
    ece_vsb = []
    ece_gpc = []

    nll_ts = []
    nll_bt = []
    nll_bt_MAP = []
    nll_vs = []
    nll_vsb = []
    nll_gpc = []

    acc_gpc = []
    for run in tqdm(range(kwargs['num_runs']), desc='Run'):
        ece_ts_run = []
        ece_bt_run = []
        ece_bt_MAP_run = []
        ece_vs_run = []
        ece_vsb_run = []
        ece_gpc_run = []

        nll_ts_run = []
        nll_bt_run = []
        nll_bt_MAP_run = []
        nll_vs_run = []
        nll_vsb_run = []
        nll_gpc_run = []

        acc_gpc_run = []
        for batch_size in tqdm(kwargs['batch_size'], desc='Batch sizes', leave=False):
            # Get a subset of the calibration dataset
            subset_idxs = np.random.choice(len(calibration_dataset), size=batch_size, replace=False)
            batch = Subset(calibration_dataset, subset_idxs)
            calibration_batch_loader = DataLoader(batch, batch_size=256, shuffle=False, num_workers=0)

            # Forward pass batch through model ; get logits and labels
            logits, labels = model_utils.forward_pass(model, calibration_batch_loader, kwargs['num_classes'],
                                                      device=gpu, verbose=False)
            logits = logits.to(cpu)
            labels = labels.to(cpu)

            # =============================================
            # Perform calibration with the various methods
            # =============================================

            # ----| Temperature scaling
            ts_temperature = temperature_scaling(logits, labels.long())['temperature']
            eval_probs_ts = softmax(1. / ts_temperature * eval_logits, dim=1)
            # --------| Get metrics
            ece_ts_run.append(expected_calibration_error(eval_probs_ts, eval_labels))
            nll_ts_run.append(nll(eval_probs_ts.log(), eval_labels.long()).item())

            # ----| Vector scaling
            vs_temperature = vector_scaling(logits, labels.long(), bias=False)['temperature']
            eval_probs_vs = softmax(1. / vs_temperature * eval_logits, dim=1)
            # --------| Get metrics
            ece_vs_run.append(expected_calibration_error(eval_probs_vs, eval_labels))
            nll_vs_run.append(nll(eval_probs_vs.log(), eval_labels.long()).item())

            # ----| Vector scaling with bias
            vsb_temperature = vector_scaling(logits, labels.long(), bias=True)['temperature']
            eval_probs_vsb = softmax(1. / vsb_temperature * eval_logits, dim=1)
            # --------| Get metrics
            ece_vsb_run.append(expected_calibration_error(eval_probs_vsb, eval_labels))
            nll_vsb_run.append(nll(eval_probs_vsb.log(), eval_labels.long()).item())

            # ----| Bayesian Tempering
            bt_model = BT(kwargs['prior_params_bt'], kwargs['num_classes'], **kwargs['mcmc_params'], verbose=False)
            bt_model.update(logits, labels)
            eval_probs_bt = bt_model.calibrate(eval_logits)
            # --------| Get metrics
            ece_bt_run.append(expected_calibration_error(eval_probs_bt, eval_labels))
            nll_bt_run.append(nll(eval_probs_bt.log(), eval_labels.long()).item())

            # ----| Bayesian Tempering (MAP)
            bt_MAP_temperature = bt_model.get_MAP_temperature(logits, labels)
            eval_probs_bt_MAP = softmax(1./bt_MAP_temperature * eval_logits, dim=1)
            # --------| Get metrics
            ece_bt_MAP_run.append(expected_calibration_error(eval_probs_bt_MAP, eval_labels))
            nll_bt_MAP_run.append(nll(eval_probs_bt_MAP.log(), eval_labels.long()).item())

            # ----| GP Calibration
            # !!!!! Important: If you do not set random_seed when defining gpc, their code will
            # reset the seed to zero on every iteration -- makes you get the same batch every run!
            # You need to set random_state (np.random.seed) to a different value per step (here or otherwise)
            gpc = GPCalibration(logits=True, n_classes=kwargs['num_classes'],
                                random_state=(batch_size + run)*kwargs['seed'])
            gpc.fit(logits.numpy().astype(float), labels.numpy().astype(np.int64))  # Need to do dtype conversions
            eval_probs_gpc = torch.tensor(gpc.predict_proba(eval_logits.numpy().astype(float)))
            # --------| Get metrics
            ece_gpc_run.append(expected_calibration_error(eval_probs_gpc, eval_labels))
            nll_gpc_run.append(nll(eval_probs_gpc.log(), eval_labels.long()).item())
            # Save accuracy as GPC is not accuracy preserving
            acc_gpc_run.append((eval_probs_gpc.max(dim=1)[1] == eval_labels).float().mean().item())

        ece_vs.append(ece_vs_run)
        nll_vs.append(nll_vs_run)

        ece_ts.append(ece_ts_run)
        nll_ts.append(nll_ts_run)

        ece_bt.append(ece_bt_run)
        nll_bt.append(nll_bt_run)

        ece_bt_MAP.append(ece_bt_MAP_run)
        nll_bt_MAP.append(nll_bt_MAP_run)

        ece_vsb.append(ece_vsb_run)
        nll_vsb.append(nll_vsb_run)

        ece_gpc.append(ece_gpc_run)
        nll_gpc.append(nll_gpc_run)

        acc_gpc.append(acc_gpc_run)

        # Save after every run in case of crash / early end
        pd.DataFrame(data=ece_vs, columns=kwargs['batch_size']).to_csv('ece_vs.csv')
        pd.DataFrame(data=nll_vs, columns=kwargs['batch_size']).to_csv('nll_vs.csv')

        pd.DataFrame(data=ece_ts, columns=kwargs['batch_size']).to_csv('ece_ts.csv')
        pd.DataFrame(data=nll_ts, columns=kwargs['batch_size']).to_csv('nll_ts.csv')

        pd.DataFrame(data=ece_bt, columns=kwargs['batch_size']).to_csv('ece_bt.csv')
        pd.DataFrame(data=nll_bt, columns=kwargs['batch_size']).to_csv('nll_bt.csv')

        pd.DataFrame(data=ece_bt_MAP, columns=kwargs['batch_size']).to_csv('ece_bt_MAP.csv')
        pd.DataFrame(data=nll_bt_MAP, columns=kwargs['batch_size']).to_csv('nll_bt_MAP.csv')

        pd.DataFrame(data=ece_vsb, columns=kwargs['batch_size']).to_csv('ece_vsb.csv')
        pd.DataFrame(data=nll_vsb, columns=kwargs['batch_size']).to_csv('nll_vsb.csv')

        pd.DataFrame(data=ece_gpc, columns=kwargs['batch_size']).to_csv('ece_gpc.csv')
        pd.DataFrame(data=nll_gpc, columns=kwargs['batch_size']).to_csv('nll_gpc.csv')

        pd.DataFrame(data=acc_gpc, columns=kwargs['batch_size']).to_csv('acc_gpc.csv')

    pd.DataFrame(data=ece_vs, columns=kwargs['batch_size']).to_csv('ece_vs.csv')
    pd.DataFrame(data=nll_vs, columns=kwargs['batch_size']).to_csv('nll_vs.csv')

    pd.DataFrame(data=ece_ts, columns=kwargs['batch_size']).to_csv('ece_ts.csv')
    pd.DataFrame(data=nll_ts, columns=kwargs['batch_size']).to_csv('nll_ts.csv')

    pd.DataFrame(data=ece_bt, columns=kwargs['batch_size']).to_csv('ece_bt.csv')
    pd.DataFrame(data=nll_bt, columns=kwargs['batch_size']).to_csv('nll_bt.csv')

    pd.DataFrame(data=ece_bt_MAP, columns=kwargs['batch_size']).to_csv('ece_bt_MAP.csv')
    pd.DataFrame(data=nll_bt_MAP, columns=kwargs['batch_size']).to_csv('nll_bt_MAP.csv')

    pd.DataFrame(data=ece_vsb, columns=kwargs['batch_size']).to_csv('ece_vsb.csv')
    pd.DataFrame(data=nll_vsb, columns=kwargs['batch_size']).to_csv('nll_vsb.csv')

    pd.DataFrame(data=ece_gpc, columns=kwargs['batch_size']).to_csv('ece_gpc.csv')
    pd.DataFrame(data=nll_gpc, columns=kwargs['batch_size']).to_csv('nll_gpc.csv')

    pd.DataFrame(data=acc_gpc, columns=kwargs['batch_size']).to_csv('acc_gpc.csv')

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
