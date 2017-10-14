#!/usr/bin/python3

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = 'data/fashion'
IMAGE_SIZE = 28

def load_dataset():
    data = input_data.read_data_sets(DATASET_PATH, one_hot=True)
    return data

def get_hparams(hparams_arg):
    hparams = tf.contrib.training.HParams(learning_rate=0.001, conv1_depth=32, conv2_depth=128,
                                          dense_layer_units=1024, batch_size=128, 
                                          keep_prob=0.5, num_epochs=1)
    if hparams_arg:
        hparams.parse(hparams_arg)
    return hparams

def shuffle_dataset(images, labels):
    num_examples = images.shape[0]
    permutation = list(np.random.permutation(num_examples))
    return (images[permutation, :], labels[permutation, :])

