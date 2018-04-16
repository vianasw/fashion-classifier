#!/usr/bin/python3

from models.model import Model
import tensorflow as tf


class LeNet(Model):
    """CNN model for image classification

    The CNN Architecture is basically LeNet 5 slightly modified. Hidden units
    numbers may vary but baseline is as follows:
            * Convolutional Layer #1: Applies 5x5 filters
              (extracting 5x5-pixel subregions), with ReLU activation function
            * Pooling Layer #1: Performs max pooling with a 2x2 filter
              and stride of 2 (which specifies that pooled regions do not
              overlap)
            * Convolutional Layer #2: Applies 5x5 filters, with ReLU
              activation function
            * Pooling Layer #2: Again, performs max pooling with a 2x2 filter
              and stride of 2
            * Dense Layer #1: Fully connected layer
            * Dense Layer #2 (Logits Layer): 10 hidden units, one for each
              target class (0–9)
    Arguments:
        hparams: tf.contrib.training.HParams, hyper parameters
        image_size: Integer, specifies the width and height of the images from
            the datasets
        num_channels: Integer, number of channels of the image (1 for
            greyscale, 3 for RGB)
        num_classes: Integer, number of classes to predict
    """
    def __init__(self, hparams, image_size, num_channels, num_classes):
        self.hparams = hparams
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self._setup_model_params()
        self.init = None

    def X(self):
        return self._X

    def Y(self):
        return self._Y

    def optimizer(self, global_step=None):
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate).minimize(
                self.cost, global_step=global_step)
        return optimizer

    def forward_propagation(self, keep_prob, training=False):
        """Defines feed forward computation graph.

        Arguments:
            X: Input dataset placeholder, of shape (batch_size, image_size,
                image_size, num_channels).

        Returns:
            fc2: The output of the last linear unit
        """

        conv1 = self.conv_layer(input=self._X, size_in=self.num_channels,
                                size_out=self.hparams.conv1_depth,
                                patch_size=self.hparams.patch_size, conv_stride=1,
                                name='conv1')

        conv2 = self.conv_layer(input=conv1, size_in=self.hparams.conv1_depth,
                                size_out=self.hparams.conv2_depth,
                                patch_size=self.hparams.patch_size,
                                conv_stride=1, name='conv2')

        shape = conv2.get_shape().as_list()
        fc1_size_in = shape[1] * shape[2] * shape[3]
        flattened = tf.reshape(conv2, [-1, fc1_size_in])
        fc1 = tf.nn.relu(
            self.fully_connected_layer(flattened, size_in=fc1_size_in,
                                       size_out=self.hparams.dense_layer_units,
                                       name='fc1')
        )

        self.embedding_input = fc1

        fc1_dropout = self.dropout(fc1, keep_prob, training)

        fc2 = self.fully_connected_layer(fc1_dropout,
                                         size_in=self.hparams.dense_layer_units,
                                         size_out=self.num_classes,
                                         name='fc2')

        return fc2

    def _setup_model_params(self):
        tf.reset_default_graph()
        self._X, self._Y = self._create_placeholders()
        self.regularization = 0
        logits = self.forward_propagation(self.hparams.keep_prob, training=True)
        self.cost = self._compute_cost(logits, self._Y)

    def _create_placeholders(self):
        X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size,
                           self.num_channels], name='X')
        Y = tf.placeholder(tf.float32, [None, self.num_classes], name='Y')
        return X, Y

    def _compute_cost(self, logits, labels):
        with tf.name_scope('xent'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels) + self.regularization)
            tf.summary.scalar("cost", self.cost)
        return self.cost
