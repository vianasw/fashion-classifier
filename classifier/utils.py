#!/usr/bin/python3

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = 'data/fashion'
IMAGE_SIZE = 28
def load_dataset():
    data = input_data.read_data_sets(DATASET_PATH, one_hot=True)
    return data

def show_random_image(dataset):
    batch = dataset.train.next_batch(1) 
    import matplotlib.pyplot as plt
    x, y = batch
    plt.imshow(x.reshape([IMAGE_SIZE, IMAGE_SIZE]))
    plt.gray()
    plt.show()

def plot_costs(costs, title):
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per tens)")
    plt.title(title)
    plt.show()

def batch_accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
