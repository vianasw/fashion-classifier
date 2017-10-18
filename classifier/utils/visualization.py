#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np


def show_image(image, width, height):
    plt.imshow(image.reshape([width, height]))
    plt.gray()
    plt.show()


def create_sprite_image(images, save_path):
    """Creates and save a sprite image consisting of images passed as argument.

    Arguments:
        images: numpy like array with shape [num_images, height, width].
        save_path: path to save sprite image.

    Returns:
        sprite: sprite image consisting of the images passed as images.
    """
    if isinstance(images, list):
        images = np.array(images)
    height = images.shape[1]
    width = images.shape[2]
    num_images = int(np.ceil(np.sqrt(images.shape[0])))

    sprite = np.ones((height * num_images, width * num_images))

    for i in range(num_images):
        for j in range(num_images):
            img_idx = i * num_images + j
            if img_idx < images.shape[0]:
                image = images[img_idx]
                sprite[i * height:(i + 1) * height,
                       j * width:(j + 1) * width] = image

    plt.imsave(save_path, sprite, cmap='gray')
    return sprite


def invert_grayscale(images):
    """ Makes black white, and white black """
    return 1 - images
