import pathlib
import yaml
from utils import utils, model_utils, data_utils

""" This file runs a non-sequential experiment from a given config yml file. 
"""


def main():
    config_file = ''  # TODO: Set config file

    with open(config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Set RNG seed
    if 'seed' in config:
        utils.set_seed(config['seed'])

    # Load our pre-trained model
    model = model_utils.load_trained_model(config['model'], config['train_set'])
    # Get a fixed calibration / evaluation set
    calibration_dataset, eval_dataset = data_utils.get_cal_eval_split(config['test_set'], config['num_eval'])


if __name__ == '__main__':
    main()
