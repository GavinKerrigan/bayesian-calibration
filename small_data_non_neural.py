# Standard libs
import yaml
import numpy as np
import time
import pandas as pd
from tqdm.notebook import tqdm
# Torch utils
from torch.nn.functional import softmax
from torch.nn import NLLLoss
# Custom utils
from utils import utils
from utils.metrics import *
# Calibration methods
from calibration_methods.bayesian_tempering import BayesianTemperingCalibrator as BT
from calibration_methods.maximum_likelihood import vector_scaling, temperature_scaling
from pycalib.calibration_methods import GPCalibration
# Sklearn utils
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

""" This file runs a non-sequential experiment from a given config .yml file. 
Specifically for non-neural models. 
"""


# Helper function to map probabilities to unnormalized logits
def probs_to_logits(probs):
    eps = 1e-30
    clipped_probs = np.clip(probs, eps, 1)
    logits = torch.from_numpy(np.log(clipped_probs))
    return logits


def preprocess_adult():
    dataset = fetch_openml('adult', version=2)
    X, y = dataset['data'], dataset['target']

    # Encode labels with ints
    enc = LabelEncoder()
    enc.fit(y)
    y = enc.transform(y).flatten()

    # Store into DF for easy pre-processing
    df = pd.DataFrame(X, columns=dataset['feature_names'])
    df['Label'] = y

    # Pre-processing
    df = df.dropna(axis=0)  # Drop NaNs
    drop_columns = ['fnlwgt', 'education', 'native-country', 'relationship']
    df = df.drop(columns=drop_columns)
    categorical_columns = ['marital-status', 'occupation', 'race', 'sex']
    df = pd.get_dummies(df, columns=categorical_columns)

    y = df.pop('Label')

    return df, y


def preprocess_car():
    dataset = fetch_openml('car', version=3)
    X, y = dataset['data'], dataset['target']

    # Encode labels
    enc = LabelEncoder()
    enc.fit(y)
    y = enc.transform(y).flatten()

    return X, y


def load_dataset(name, num_eval, num_cal, seed):
    if name == 'adult':
        X, y = preprocess_adult()
    elif name == 'car':
        X, y = preprocess_car()
    elif name == 'mnist':
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        X, y = mnist['data'], mnist['target'].astype(int)

    num_datapoints = X.shape[0]
    # Split into train / other
    assert num_datapoints > num_eval + num_cal, 'You requested too many eval/cal datapoints; The dataset only has {} ' \
                                                'points.'.format(num_datapoints)
    split_size = 1. * (num_eval + num_cal) / num_datapoints
    train_data, split_data, train_labels, split_labels = train_test_split(X, y, test_size=split_size, random_state=seed)
    # Split other into eval / cal
    split_size = 1. * num_cal / (num_cal + num_eval)
    eval_data, cal_data, eval_labels, cal_labels = train_test_split(split_data, split_labels,
                                                                    test_size=split_size, random_state=seed)

    return train_data, cal_data, eval_data, train_labels, cal_labels, eval_labels


def run_experiment(model, dataset, num_eval, num_cal, seed, out_path='', **kwargs):
    t0 = time.time()
    nll = NLLLoss()

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

    # ---- Load data, fit model, get logits/labels.
    # ---- NB: This enforces your training set / calibration set / evaluation set to be fixed
    #       for all runs, and the only source of stochasticity is in the batches used for calibration.
    print('Loading data')
    train_data, cal_data, eval_data, train_labels, cal_labels, eval_labels = load_dataset(
        dataset, num_eval, num_cal, seed)
    print('Fitting model')
    model.fit(train_data, train_labels)
    # Forward pass model on eval/cal set
    eval_probs = model.predict_proba(eval_data)
    eval_logits = probs_to_logits(eval_probs)
    # Evaluate model on calibration dataset ; get logits
    cal_probs = model.predict_proba(cal_data)
    cal_logits = probs_to_logits(cal_probs)
    # Map data to tensor
    eval_labels = torch.tensor(eval_labels)
    cal_labels = torch.tensor(cal_labels)

    # Evaluate with no calibration
    ece_nocal = expected_calibration_error(softmax(eval_logits, dim=1), eval_labels)
    nll_nocal = nll(eval_logits, eval_labels.long())
    acc_nocal = (torch.from_numpy(eval_probs).max(dim=1)[1] == eval_labels).float().mean().item()

    with open('metrics_nocal.txt', 'w') as f:
        f.write('ECE no calibration: {:.4f}\n'.format(ece_nocal))
        f.write('NLL no calibration: {:.4f}'.format(nll_nocal))
        f.write('Acc no calibration: {:.4f}'.format(acc_nocal))

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
            subset_idxs = np.random.choice(num_cal, size=batch_size, replace=False)
            cal_logits_batch = cal_logits[subset_idxs, :]
            cal_labels_batch = cal_labels[subset_idxs]

            # =============================================
            # Perform calibration with the various methods
            # =============================================

            # Aliasing for convenience
            logits = cal_logits_batch.clone().detach()
            labels = cal_labels_batch.clone().detach()

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
            eval_probs_bt_MAP = softmax(1. / bt_MAP_temperature * eval_logits, dim=1)
            # --------| Get metrics
            ece_bt_MAP_run.append(expected_calibration_error(eval_probs_bt_MAP, eval_labels))
            nll_bt_MAP_run.append(nll(eval_probs_bt_MAP.log(), eval_labels.long()).item())

            # ----| GP Calibration
            # !!!!! Important: If you do not set random_seed when defining gpc, their code will
            # reset the seed to zero on every iteration -- makes you get the same batch every run!
            # You need to set random_state (np.random.seed) to a different value per step (here or otherwise)
            gpc = GPCalibration(logits=True, n_classes=kwargs['num_classes'],
                                random_state=(batch_size + run) * seed)
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
        pd.DataFrame(data=ece_vs, columns=kwargs['batch_size']).to_csv(out_path + 'ece_vs.csv')
        pd.DataFrame(data=nll_vs, columns=kwargs['batch_size']).to_csv(out_path + 'nll_vs.csv')

        pd.DataFrame(data=ece_ts, columns=kwargs['batch_size']).to_csv(out_path + 'ece_ts.csv')
        pd.DataFrame(data=nll_ts, columns=kwargs['batch_size']).to_csv(out_path + 'nll_ts.csv')

        pd.DataFrame(data=ece_bt, columns=kwargs['batch_size']).to_csv(out_path + 'ece_bt.csv')
        pd.DataFrame(data=nll_bt, columns=kwargs['batch_size']).to_csv(out_path + 'nll_bt.csv')

        pd.DataFrame(data=ece_bt_MAP, columns=kwargs['batch_size']).to_csv(out_path + 'ece_bt_MAP.csv')
        pd.DataFrame(data=nll_bt_MAP, columns=kwargs['batch_size']).to_csv(out_path + 'nll_bt_MAP.csv')

        pd.DataFrame(data=ece_vsb, columns=kwargs['batch_size']).to_csv(out_path + 'ece_vsb.csv')
        pd.DataFrame(data=nll_vsb, columns=kwargs['batch_size']).to_csv(out_path + 'nll_vsb.csv')

        pd.DataFrame(data=ece_gpc, columns=kwargs['batch_size']).to_csv(out_path + 'ece_gpc.csv')
        pd.DataFrame(data=nll_gpc, columns=kwargs['batch_size']).to_csv(out_path + 'nll_gpc.csv')

        pd.DataFrame(data=acc_gpc, columns=kwargs['batch_size']).to_csv(out_path + 'acc_gpc.csv')

    pd.DataFrame(data=ece_vs, columns=kwargs['batch_size']).to_csv(out_path + 'ece_vs.csv')
    pd.DataFrame(data=nll_vs, columns=kwargs['batch_size']).to_csv(out_path + 'nll_vs.csv')

    pd.DataFrame(data=ece_ts, columns=kwargs['batch_size']).to_csv(out_path + 'ece_ts.csv')
    pd.DataFrame(data=nll_ts, columns=kwargs['batch_size']).to_csv(out_path + 'nll_ts.csv')

    pd.DataFrame(data=ece_bt, columns=kwargs['batch_size']).to_csv(out_path + 'ece_bt.csv')
    pd.DataFrame(data=nll_bt, columns=kwargs['batch_size']).to_csv(out_path + 'nll_bt.csv')

    pd.DataFrame(data=ece_bt_MAP, columns=kwargs['batch_size']).to_csv(out_path + 'ece_bt_MAP.csv')
    pd.DataFrame(data=nll_bt_MAP, columns=kwargs['batch_size']).to_csv(out_path + 'nll_bt_MAP.csv')

    pd.DataFrame(data=ece_vsb, columns=kwargs['batch_size']).to_csv(out_path + 'ece_vsb.csv')
    pd.DataFrame(data=nll_vsb, columns=kwargs['batch_size']).to_csv(out_path + 'nll_vsb.csv')

    pd.DataFrame(data=ece_gpc, columns=kwargs['batch_size']).to_csv(out_path + 'ece_gpc.csv')
    pd.DataFrame(data=nll_gpc, columns=kwargs['batch_size']).to_csv(out_path + 'nll_gpc.csv')

    pd.DataFrame(data=acc_gpc, columns=kwargs['batch_size']).to_csv(out_path + 'acc_gpc.csv')

    t1 = time.time()
    print('\n\n\nFinished: runtime (s): {:.2f}'.format(t1 - t0))


def run_from_config(config_fpath):
    with open(config_fpath, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Set RNG seed
    if 'seed' in config:
        utils.set_seed(config['seed'])

    model_dict = {'RandomForest': RandomForestClassifier(),
                  'AdaBoost': AdaBoostClassifier(),
                  'GaussianNB': GaussianNB(),
                  'LogisticRegression': LogisticRegression()}
    model = model_dict[config.pop('model')]
    dataset = config.pop('dataset')
    out_path = config.pop('out_path')

    return run_experiment(model, dataset, config.pop('num_eval'), config.pop('num_cal'), config.pop('seed'),
                          out_path=out_path, **config)
