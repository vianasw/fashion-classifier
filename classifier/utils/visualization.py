#!/usr/bin/python3

import matplotlib.pyplot as plt


def show_image(image, width, height):
    plt.imshow(image.reshape([width, height]))
    plt.gray()
    plt.show()

