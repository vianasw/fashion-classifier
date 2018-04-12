#!/usr/bin/python3

from models.model import Model
import tensorflow as tf


class VGG(Model):

    def __init__(self, hparams, image_size, num_channels, num_classes):
        self.hparams = hparams
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self._setup_model_params()

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
        conv1 = self.conv_layer(input=self._X, size_in=self.num_channels,
                                size_out=64, patch_size=3, conv_stride=1,
                                name='conv1', num_convolutions=3)
        
        conv2 = self.conv_layer(input=conv1, size_in=64,
                                size_out=128, patch_size=3, conv_stride=1,
                                name='conv2', num_convolutions=3)

        shape = conv2.get_shape().as_list()
        fc1_size_in = shape[1] * shape[2] * shape[3]
        flattened = tf.reshape(conv2, [-1, fc1_size_in])
        flattened_dropout = self.dropout(flattened, keep_prob, training)
        fc1 = tf.nn.relu(
            self.fully_connected_layer(flattened_dropout, size_in=fc1_size_in,
                                       size_out=4096,
                                       name='fc1')
        )

        fc1_dropout = self.dropout(fc1, keep_prob, training)

        fc2 = self.fully_connected_layer(fc1_dropout,
                                         size_in=4096,
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
