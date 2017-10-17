#!/usr/bin/python3

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


DATASET_PATH = 'data/fashion'

def load_dataset():
    """Loads fashion dataset

    Returns:
        Datasets object
    """
    data = input_data.read_data_sets(DATASET_PATH, one_hot=True)
    return data

def get_hparams(hparams_str):
    """Parses hparams_str to HParams object.

    Arguments:
        hparams_str: String of comma separated param=value pairs.

    Returns:
        hparams: tf.contrib.training.HParams object from hparams_str. If
            hparams_str is None, then a default HParams object is returned.
    """
    hparams = tf.contrib.training.HParams(learning_rate=0.001, conv1_depth=32, conv2_depth=128,
                                          dense_layer_units=1024, batch_size=128,
                                          keep_prob=0.5, lambd=0.01, num_epochs=1, augment_percent=0.0)
    if hparams_str:
        hparams.parse(hparams_str)
    return hparams

def shuffle_dataset(images, labels):
    num_examples = images.shape[0]
    permutation = list(np.random.permutation(num_examples))
    return (images[permutation, :], labels[permutation, :])

