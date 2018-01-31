#!/usr/bin/python3

from abc import ABC, abstractmethod
import tensorflow as tf

class Model(ABC):

    @abstractmethod
    def forward_propagation(self, keep_prob, training=True):
        """Performs forward propagation
        Arguments:
            keep_prob: A floating point scalar Tensor used during training for dropout operation
            training: Boolean, indicates if it's in training mode
        """
        pass

    @property
    @abstractmethod
    def X(self):
        """Placeholder for a tensor to feed samples to the model"""
        pass

    @property
    @abstractmethod
    def Y(self):
        """Placeholder for a tensor to feed labels to the model"""
        pass

    def logits(self):
        """Tensor after performing forward propagation operation

        Returns:
            Tensor after performing forward propagation operation
        """
        return self.forward_propagation(keep_prob=1.0, training=False)

    def evaluate(self, X):
        return tf.nn.softmax(self.logits(X))

    def accuracy(self, logits):
        """Calculates accuracy given logits

        Arguments:
            logits: tensor returned by performing forward propagation operation

        Returns:
            Accuracy tensor
        """
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1),
                                          tf.argmax(self.Y(), 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            return accuracy

    def feed_dict(self, X, Y):
        """Returns feed dictionary to be used during training"""
        return {self.X(): X, self.Y(): Y}

    def conv_layer(self, input, size_in, size_out, patch_size, conv_stride,
                    name='conv'):
        """Creates convolutional layer.

        Arguments:
            input: Tensor, placeholder tensor to apply convolution operation
            size_in: Integer, number of input neurons
            size_out: Integer, number of output neurons
            patch_size: Integer, size of the convolutional patch to apply
            conv_stride: Integer, number of strides to apply
            name: String, name used for variable_scope
        Returns:
            Tensor
        """
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
        """Creates fully conntected layer.

        Arguments:
            input: Tensor, placeholder tensor
            size_in: Integer, number of input neurons
            size_out: Integer, number of output neurons
            name: String, name used for variable_scope
        Returns:
            Tensor
        """
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

    def dropout(self, input, keep_prob, training=False, name='dropout'):
        """Returns dropout tensor if training parameter is True

        Arguments:
            input: Tensor, placeholder tensor
            keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept
            training: Boolean, indicates if it's in training mode
            name: String, name to use in name_scope (optional)

        Returns:
            Tensor
        """
        with tf.name_scope(name):
            if training:
                keep_prob = tf.constant(keep_prob)
            else:
                keep_prob = tf.constant(1.0)
            return tf.nn.dropout(input, keep_prob)

