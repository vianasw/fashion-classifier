#!/usr/bin/python3

from utils import load_dataset
from itertools import product
from random import shuffle
from fashion_classifier import FashionClassifier
import tensorflow as tf
import json
import argparse
import sys
import os


DEFAULT_LOGDIR = '/tmp/fashion-classifier/hparam_search/'
FLAGS = None

def hparam_product(param_grid):
    return [dict(zip(param_grid, param_values)) for param_values in product(*param_grid.values())]

def hparam_string(param_grid):
    return ','.join(['{0}={1}'.format(k, v) for k, v in param_grid.items()])

def main(_):
    dataset = load_dataset()
    X_train, Y_train = dataset.train.images, dataset.train.labels
    X_test, Y_test = dataset.test.images, dataset.test.labels

    if FLAGS.hparams_path:
        with open(FLAGS.hparams_path) as hparams_json:
            param_grid = json.load(hparams_json)
    else:
        param_grid = {
            "conv1_depth": [16, 32, 64],
            "conv2_depth": [64, 128],
            "dense_layer_units": [1024, 2048],
            "batch_size": [64, 128],
            "lambd": [0.5],
            "num_epochs": [5]
        }

    hparams_subset = hparam_product(param_grid)
    shuffle(hparams_subset)
    grid_size = min(len(hparams_subset), FLAGS.grid_size)
    hparams_subset = hparams_subset[:grid_size]

    for hparams in hparams_subset:
        print('***************************************')
        print('Training with: {0}\n\n'.format(hparam_string(hparams)))

        checkpoint_subdir = hparam_string(hparams)
        conv_depths = [hparams['conv1_depth'], hparams['conv2_depth']]
        dense_layer_units = hparams['dense_layer_units']
        batch_size = hparams['batch_size']
        lambd = hparams['lambd']
        num_epochs = hparams['num_epochs']

        log_dir = os.path.join(FLAGS.logdir, checkpoint_subdir)
        fashion_classifier = FashionClassifier(X_train, Y_train, X_test, Y_test,
                                                image_size=28, num_channels=1,
                                                num_classes=10,
                                                log_dir=log_dir)

        fashion_classifier.model(padding='SAME', patch_size=5,
                                conv_depths=conv_depths, dense_layer_units=dense_layer_units,
                                learning_rate=0.001, batch_size=batch_size,
                                lambd=lambd)

        fashion_classifier.train_and_evaluate(num_epochs=num_epochs,
                                              resume_training=False,
                                              print_cost=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, default=DEFAULT_LOGDIR,
                        help='Store log/model files.')
    parser.add_argument('--hparams_path', type=str, default=None,
                        help='Path to json file with hparams.')
    parser.add_argument('--grid_size', type=int, default=9,
                        help='Size of the hyperparams grid to explore')

    FLAGS, _ = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]])
