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


def shuffle_dataset(images, labels):
    num_examples = images.shape[0]
    permutation = list(np.random.permutation(num_examples))
    return (images[permutation, :], labels[permutation, :])
