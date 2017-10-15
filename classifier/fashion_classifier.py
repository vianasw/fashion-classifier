#!/usr/bin/python3

from utils.misc_utils import load_dataset, get_hparams, shuffle_dataset
from utils.data_augmentation import augment_data
import argparse
import tensorflow as tf
import numpy as np
import os
import sys

FLAGS = None
DEFAULT_LOGDIR = '/tmp/fashion-classifier/logdir/'


class FashionClassifier:
    """Classifier for the Fashion-MNIST Zalando's dataset [1].

    Convolutional neural network to classify Zalando's article images from 10
    different classes (i.e. trouser, dress, coat, bag, etc).
    [1]: https://github.com/zalandoresearch/fashion-mnist

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
            * Dense Layer #1: Hidden units, with dropout regularization
            * Dense Layer #2 (Logits Layer): 10 hidden units, one for each
              target class

    Arguments:
        X_train: numpy array of training examples of shape [num_examples,
            image_size * image_size]
        Y_train: np array of training labels of shape [num_examples, 1]
        X_test: np array of test examples of shape [num_examples,
            image_size * image_size]
        Y_test: np array of test labels of shape [num_examples, 1]
        image_size: Integer, specifies the width and height of the images from
            the datasets.
        num_channels: Integer, number of channels of the image (1 for
            greyscale, 3 for RGB).
        num_classes: Integer, number of classes to predict.
        log_dir: Path to save logs for Tensorboard and checkpoint files
        checkpoint_filename: name of the checkpoint file, default: model.ckpt
    """
    def __init__(self, X_train, Y_train, X_test, Y_test, image_size,
                 num_channels, num_classes, log_dir,
                 checkpoint_filename='model.ckpt'):

        self.padding = 'SAME'
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X = None
        self.Y = None
        self.cost = None
        self.optimizer = None
        self.logits = None
        self.batch_size = 64
        self.writer = tf.summary.FileWriter(log_dir)
        self.log_dir = log_dir
        self.checkpoint_filename = checkpoint_filename

    def model(self, padding, patch_size, conv_depths, dense_layer_units,
              learning_rate, batch_size, keep_prob):
        """Defines the CNN model.

        Creates the computational graph for the CNN and set all hyperparameters

        Arguments:
            padding: String, convolution padding ('SAME' or 'VALID').
            patch_size: Integer, patch size to apply in conv layers
            conv_depths: Array of integers with the depths of the convolution
                layers.
            dense_layer_units: Integer, number of hidden units in the dense
                layer.
            learning_rate: Float, starter learning rate hyperparameter
                for learning rate decay.
            batch_size: Integer, size of the batch to process in each training
                step.
            keep_prob: Float, dropout keep probability (from 0.0 to 1.0) to
                apply during training.
        """

        tf.reset_default_graph()

        self.padding = padding
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.conv_depths = conv_depths
        self.dense_layer_units = dense_layer_units

        self.X, self.Y = self._create_placeholders()
        # this is to display some image examples on Tensorboard
        tf.summary.image('input', self.X, 3)

        self.logits = self._forward_propagation(
                self.X, keep_prob, training=True)
        self.cost = self._compute_cost(self.logits, self.Y)

        self.global_step = tf.Variable(
                0, dtype=tf.int32, trainable=False, name='global_step')
        self.current_step = tf.Variable(
                0, dtype=tf.int32, trainable=False, name='current_step')

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                    self.cost, global_step=self.global_step)

    def train_and_evaluate(self, num_epochs, resume_training=False,
                           print_cost=False):
        """Performs training for the CNN defined with the model method.

        Arguments:
            num_epochs: Integer, number of epochs to perform training.
            resume_training: Boolean, if True restores tensor values from
            checkpoint.
            print_cost: Boolean, if True, prints costs every some iterations.
        """
        init = tf.global_variables_initializer()
        eval_logits = self._forward_propagation(self.X, keep_prob=1.0,
                                                training=False)
        train_prediction = tf.nn.softmax(eval_logits)
        accuracy = self._accuracy(eval_logits)

        num_examples = self.X_train.shape[0]
        shuffled_X, shuffled_Y = shuffle_dataset(self.X_train, self.Y_train)

        summ_op = tf.summary.merge_all()

        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()

            if resume_training:
                self._restore_last_checkpoint(session, saver)

            self.writer.add_graph(session.graph)

            num_minibatches = int(num_examples / self.batch_size)
            current_epoch = self.global_step.eval() // (num_minibatches)
            current_step = self.current_step.eval()
            for epoch in range(current_epoch, num_epochs):
                epoch_cost = 0
                for step in range(current_step, num_minibatches):
                    (minibatch_X, minibatch_Y) = self._next_batch(step)

                    _, minibatch_cost, predictions = session.run(
                            [self.optimizer, self.cost, train_prediction],
                            feed_dict={self.X: minibatch_X,
                                       self.Y: minibatch_Y}
                        )

                    epoch_cost += minibatch_cost / num_minibatches

                    self._log_progress(
                            session, summ_op, accuracy, minibatch_cost,
                            num_minibatches, minibatch_X, minibatch_Y, epoch,
                            step, print_cost)

                    self._save_checkpoint(session, saver, step)
                current_step = 0

                # Handling the end case (last mini-batch < batch_size)
                if num_examples % num_minibatches != 0:
                    (minibatch_X, minibatch_Y) = self._last_batch(
                                                    num_minibatches)
                    _, minibatch_cost, predictions = session.run(
                            [self.optimizer, self.cost, train_prediction],
                            feed_dict={self.X: minibatch_X,
                                       self.Y: minibatch_Y}
                        )

                    epoch_cost += minibatch_cost / num_minibatches

                if print_cost:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))

            self._evaluate(eval_logits)

    def load_and_evaluate(self):
        """Loads model from last checkpoint stored in log_dir."""
        init = tf.global_variables_initializer()
        eval_logits = self._forward_propagation(self.X, keep_prob=1.0,
                                                training=False)

        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            self._restore_last_checkpoint(session, saver)
            self._evaluate(eval_logits)

    def _create_placeholders(self):
        X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size,
                           self.num_channels], name='X')
        Y = tf.placeholder(tf.float32, [None, self.num_classes], name='Y')
        return X, Y

    def _forward_propagation(self, X, keep_prob=1.0, training=False):
        """Defines feed forward computation graph.

        Arguments:
            X: Input dataset placeholder, of shape (batch_size, image_size,
                image_size, num_channels).
            parameters: Dictionary containing the initialized weights and
                biasses.
            keep_prob: Float, dropout probability hyperparameter.
            training: Boolean, used to apply dropout regularization during
                training.

        Returns:
            fc2: The output of the last linear unit
        """

        conv1 = self._conv_layer(input=X, size_in=self.num_channels,
                                 size_out=self.conv_depths[0],
                                 patch_size=self.patch_size, conv_stride=1,
                                 name='conv1')

        conv2 = self._conv_layer(input=conv1, size_in=self.conv_depths[0],
                                 size_out=self.conv_depths[1],
                                 patch_size=self.patch_size,
                                 conv_stride=2, name='conv2')

        shape = conv2.get_shape().as_list()
        fc1_size_in = shape[1] * shape[2] * shape[3]
        flattened = tf.reshape(conv2, [-1, fc1_size_in])
        fc1 = self._fully_connected_layer(flattened, size_in=fc1_size_in,
                                          size_out=self.dense_layer_units,
                                          name='fc1')

        fc1_dropout = self._dropout(fc1, keep_prob, training)

        fc2 = self._fully_connected_layer(fc1_dropout,
                                          size_in=self.dense_layer_units,
                                          size_out=self.num_classes,
                                          name='fc2')

        return fc2

    def _conv_layer(self, input, size_in, size_out, patch_size, conv_stride,
                    name='conv'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                    'W',
                    [patch_size, patch_size, size_in, size_out],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                    'b', [size_out],
                    initializer=tf.zeros_initializer())

            conv = tf.nn.conv2d(
                    input, W, strides=[1, conv_stride, conv_stride, 1],
                    padding=self.padding)
            act = tf.nn.relu(conv + b)
            pool = tf.nn.max_pool(
                    act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                    padding=self.padding)

            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return pool

    def _fully_connected_layer(self, input, size_in, size_out, name='fc'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                    'W', [size_in, size_out],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                    'b', [size_out], initializer=tf.zeros_initializer())

            act = tf.matmul(input, W) + b

            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)

            return act

    def _dropout(self, X, keep_prob, training=False, name='dropout'):
        with tf.name_scope(name):
            if training:
                keep_prob = tf.constant(keep_prob)
            else:
                keep_prob = tf.constant(1.0)

            return tf.nn.dropout(X, keep_prob)

    def _compute_cost(self, logits, labels):
        with tf.name_scope('xent'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
            tf.summary.scalar("cost", cost)
        return cost

    def _reformat(self, dataset):
        return dataset.reshape([-1, self.image_size, self.image_size,
                               self.num_channels])

    def _evaluate(self, logits):
        with tf.name_scope('accuracy'):
            accuracy = self._accuracy(logits)
            print("Train Accuracy:", accuracy.eval(
                {self.X: self._reformat(self.X_train), self.Y: self.Y_train}))
            print("Test Accuracy:", accuracy.eval(
                {self.X: self._reformat(self.X_test), self.Y: self.Y_test}))

    def _next_batch(self, step):
        offset = (step * self.batch_size) % (self.X_train.shape[0] -
                                             self.batch_size)
        minibatch_X = self.X_train[offset:(offset + self.batch_size), :]
        minibatch_Y = self.Y_train[offset:(offset + self.batch_size), :]
        minibatch_X = self._reformat(minibatch_X)
        return minibatch_X, minibatch_Y

    def _last_batch(self, num_minibatches):
        offset = num_minibatches * self.batch_size
        minibatch_X = self.X_train[offset:, :]
        minibatch_Y = self.Y_train[offset:, :]
        minibatch_X = self._reformat(minibatch_X)
        return minibatch_X, minibatch_Y

    def _accuracy(self, logits):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1),
                                          tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            return accuracy

    def _log_progress(self, session, summ, accuracy, minibatch_cost,
                      num_minibatches, minibatch_X, minibatch_Y, epoch, step,
                      print_cost):
        if step % 5 == 0:
            [batch_accuracy, s] = session.run([accuracy, summ],
                                              feed_dict={self.X: minibatch_X,
                                                         self.Y: minibatch_Y})
            self.writer.add_summary(s, (epoch * num_minibatches) + step)
            if print_cost and step % 50 == 0:
                print("Minibatch loss at step %d: %f" % (step, minibatch_cost))
                print("Minibatch accuracy: %.1f%%" % (batch_accuracy * 100))

    def _save_checkpoint(self, session, saver, step):
        session.run(tf.assign(self.current_step, step))
        if step % 50 == 0:
            checkpoint_path = os.path.join(self.log_dir,
                                           self.checkpoint_filename)

            saver.save(session, checkpoint_path, global_step=self.global_step)

    def _restore_last_checkpoint(self, session, saver):
        assert tf.gfile.Exists(os.path.join(self.log_dir, 'checkpoint'))
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        saver.restore(session, ckpt.model_checkpoint_path)


def main(_):
    dataset = load_dataset()
    X_train, Y_train = dataset.train.images, dataset.train.labels
    X_test, Y_test = dataset.test.images, dataset.test.labels

    if FLAGS.logdir:
        log_dir = FLAGS.logdir
    else:
        log_dir = DEFAULT_LOGDIR

    hparams = get_hparams(FLAGS.hparams)
    # Data augmentation: apply random horizontal flip and random crop
    if hparams.augment_percent > 0:
        X_train, Y_train = augment_data(X_train, Y_train, 28, 28, 1,
                                        hparams.augment_percent)

    fashion_classiffier = FashionClassifier(X_train, Y_train, X_test, Y_test,
                                            image_size=28, num_channels=1,
                                            num_classes=10,
                                            log_dir=log_dir)

    conv_depths = [hparams.conv1_depth, hparams.conv2_depth]
    fashion_classiffier.model(padding='SAME', patch_size=5,
                              conv_depths=conv_depths,
                              dense_layer_units=hparams.dense_layer_units,
                              learning_rate=hparams.learning_rate,
                              batch_size=hparams.batch_size,
                              keep_prob=hparams.keep_prob)

    if FLAGS.action == 'train':
        resume_training = FLAGS.resume_training is True
        fashion_classiffier.train_and_evaluate(num_epochs=hparams.num_epochs,
                                               resume_training=resume_training,
                                               print_cost=True)
    else:
        fashion_classiffier.load_and_evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('action', choices=['train', 'load'], default=None)
    parser.add_argument('--logdir', type=str, default=None,
                        help='Store log/model files.')
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training by loading last checkpoint')
    parser.add_argument('--hparams', type=str, default=None,
                        help='Comma separated list of "name=value" pairs.')

    FLAGS, _ = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]])
