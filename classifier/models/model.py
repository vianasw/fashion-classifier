#!/usr/bin/python3

from abc import ABC, abstractmethod
import tensorflow as tf

class Model(ABC):

    @abstractmethod
    def forward_propagation(self, keep_prob, training=True):
        pass

    @property
    @abstractmethod
    def X(self):
        pass

    @property
    @abstractmethod
    def Y(self):
        pass

    def logits(self):
        return self.forward_propagation(keep_prob=1.0, training=False)

    def evaluate(self, X):
        return tf.nn.softmax(self.logits(X))

    def accuracy(self, logits):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1),
                                          tf.argmax(self.Y(), 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            return accuracy

    def feed_dict(self, X, Y):
        return {self.X(): X, self.Y(): Y}

    def conv_layer(self, input, size_in, size_out, patch_size, conv_stride,
                    name='conv'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                    'W',
                    [patch_size, patch_size, size_in, size_out],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                    'b', [size_out],
                    initializer=tf.zeros_initializer())

            self.regularization += (self.hparams.lambd * tf.nn.l2_loss(W))

            conv = tf.nn.conv2d(
                    input, W, strides=[1, conv_stride, conv_stride, 1],
                    padding=self.hparams.padding)
            act = tf.nn.relu(conv + b)
            pool = tf.nn.max_pool(
                    act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding=self.hparams.padding)

            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return pool

    def fully_connected_layer(self, input, size_in, size_out, name='fc'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                    'W', [size_in, size_out],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                    'b', [size_out], initializer=tf.zeros_initializer())

            self.regularization += (self.hparams.lambd * tf.nn.l2_loss(W))
            z = tf.matmul(input, W) + b

            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", z)

            return z

    def dropout(self, X, keep_prob, training=False, name='dropout'):
        with tf.name_scope(name):
            if training:
                keep_prob = tf.constant(keep_prob)
            else:
                keep_prob = tf.constant(1.0)
            return tf.nn.dropout(X, keep_prob)

