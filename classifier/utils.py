#!/usr/bin/python3

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = 'data/fashion'
IMAGE_SIZE = 28

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
                                          lambd=0.5, num_epochs=1)
    if hparams_str:
        hparams.parse(hparams_str)
    return hparams

def show_image(image, width, height):
    plt.imshow(image.reshape([width, height]))
    plt.gray()
    plt.show()

def augment_data(images, labels, width, height, num_channels, percent):
    """Augment dataset by applying rando horizontal flip and random crop to
    a percentage of images.

    Arguments:
        images: Array of images with shape [batch_size,
            width * height * num_channels]
        labels: Array of labels with shape [batch_size]
        width: Int, width of the images.
        height: Int, height of the images.
        num_channels: Int, number of channels.
        percent: Float between 0 and 1 to indicate the percentage of images to
            apply the transformations to.

    Returns:
        (images, labels): Tuple of augmented images and labels
    """
    images_flipped, labels_flipped = random_flip_left_right(images, labels,
                                                            width, height,
                                                            num_channels,
                                                            percent)
    images_cropped, labels_cropped = random_crop(images, labels,
                                                 width, height,
                                                 num_channels, percent)

    images = np.concatenate((images, images_flipped), axis=0)
    labels = np.concatenate((labels, labels_flipped), axis=0)
    images = np.concatenate((images, images_cropped), axis=0)
    labels = np.concatenate((labels, labels_cropped), axis=0)
    num_examples = images.shape[0]
    permutation = list(np.random.permutation(num_examples))
    return images[permutation, :], labels[permutation, :]

def random_flip_left_right(images, labels, width, height, num_channels, percent):
    """Randomly flips images horizontally.

    Arguments:
        images: Array of images with shape [batch_size,
            width * height * num_channels]
        labels: Array of labels with shape [batch_size]
        width: Int, width of the images.
        height: Int, height of the images.
        num_channels: Int, number of channels.
        percent: Float between 0 and 1 to indicate the percentage of images to
            apply the transformations to.

    Returns:
        (images, labels): Tuple of flipped images and labels
    """
    images_shuffled, labels_shuffled = shuffle_dataset(images, labels)

    images_reshaped = images_shuffled.reshape([-1, width, height, num_channels])
    slice_size = int(percent * images.shape[0])
    flipped = np.fliplr(images_reshaped[:slice_size, :, :, :].T)
    flipped = flipped.T
    return flipped.reshape([-1, width * height * num_channels]), labels_shuffled[:slice_size, :]

def random_crop(images, labels, width, height, num_channels, percent):
    """Randomly crops images with a 20x20 bounding box and resize them to 28x28

    Arguments:
        images: Array of images with shape [batch_size,
            width * height * num_channels]
        labels: Array of labels with shape [batch_size]
        width: Int, width of the images.
        height: Int, height of the images.
        num_channels: Int, number of channels.
        percent: Float between 0 and 1 to indicate the percentage of images to
            apply the transformations to.

    Returns:
        (images, labels): Tuple of cropped images and labels.
    """

    images_shuffled, labels_shuffled = shuffle_dataset(images, labels)
    images_reshaped = images_shuffled.reshape([-1, width, height, num_channels])
    slice_size = int(percent * images.shape[0])
    boxes = np.repeat(np.array(
                               [[4./(height-1),
                                 4./(width-1),
                                 24./(height-1),
                                 24./(width-1)]]),
                      slice_size, axis=0)
    cropped = tf.image.crop_and_resize(images_reshaped, boxes, np.arange(slice_size), [width, height])
    cropped = tf.reshape(cropped, [-1, width * height * num_channels])
    with tf.Session() as session:
        return cropped.eval(), labels_shuffled[:slice_size, :]


def shuffle_dataset(images, labels):
    num_examples = images.shape[0]
    permutation = list(np.random.permutation(num_examples))
    return (images[permutation, :], labels[permutation, :])

