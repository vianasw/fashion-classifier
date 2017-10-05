#!/usr/bin/python3

from utils import load_dataset, plot_costs, batch_accuracy
import tensorflow as tf
import numpy as np

class FashionClassifier:
    """Classifier for the Fashion-MNIST Zalando's dataset [1].
    
    Convolutional neural network to classify Zalando's article images from 10
    different classes (i.e. trouser, dress, coat, bag, etc). 
    [1]: https://github.com/zalandoresearch/fashion-mnist

    The CNN Architecture is basically LeNet 5 slightly modified. Hidden units 
    numbers may vary but baseline is as follows:
            * Convolutional Layer #1: Applies 20 5x5 filters
              (extracting 5x5-pixel subregions), with ReLU activation function
            * Pooling Layer #1: Performs max pooling with a 2x2 filter
              and stride of 2 (which specifies that pooled regions do not overlap)
            * Convolutional Layer #2: Applies 50 5x5 filters, with ReLU activation function
            * Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
            * Dense Layer #1: 500 hidden units, with dropout regularization 
            * Dense Layer #2 (Logits Layer): 10 hidden units, one for each digit target class (0–9).

    Arguments:
      X_train: np array of training examples of shape [num_examples, image_size * image_size]
      Y_train: np array of training labels of shape [num_examples, 1]
      X_test: np array of test examples of shape [num_examples, image_size * image_size]
      Y_test: np array of test labels of shape [num_examples, 1]
      image_size: Integer, specifies the width and height of the images from 
        the datasets.
      num_channels: Integer, number of channels of the image (1 for greyscale, 
        3 for RGB).
      num_classes: Integer, number of classes to predict.
      log_dir: Path to save logs for Tensorboard
    """
    def __init__(self, X_train, Y_train, X_test, Y_test, image_size, 
            num_channels, num_classes, log_dir):

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
        self.learning_rate = 0.001
        self.writer = tf.summary.FileWriter(log_dir)

    def model(self, padding, patch_size, depth, dense_layer_units, learning_rate, batch_size, keep_prob):
        """Defines the CNN model.
        
        Creates the computational graph for the CNN and set all hyperparameters.

        Arguments:
          padding: String, convolution padding ('SAME' or 'VALID').
          depth: Integer, depth of the first convolution layer.
          dense_layer_units: Integer, number of hidden units in the dense layer.
          learning_rate: Float, learning rate hyperparameter.
          batch_size: Integer, size of the batch to process in each training step.
          keep_prob: Float, dropout keep probability (from 0.0 to 1.0) to 
            apply during training.
        """

        self.padding = padding
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.depth = depth
        self.dense_layer_units = dense_layer_units
        self.learning_rate = learning_rate

        self.X, self.Y = self._create_placeholders()
        tf.summary.image('input', self.X, 3)

        self.logits = self._forward_propagation(self.X, keep_prob, training=True)
        self.cost = self._compute_cost(self.logits, self.Y)
 
        
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


    def train(self, num_epochs, print_cost=False):
        """Performs training for the CNN defined with the model method.
        
        Peforms mini-batch gradient descent.

        Arguments:
          num_epochs: Integer, number of epochs to perform training.
          print_cost: Boolean, if True, prints costs every X iterations.
        """

        init = tf.global_variables_initializer()
        costs = []

        train_prediction = tf.nn.softmax(self.logits)

        num_examples = self.X_train.shape[0]
        permutation = list(np.random.permutation(num_examples))
        shuffled_X = self.X_train[permutation, :]
        shuffled_Y = self.Y_train[permutation, :]

        with tf.name_scope('batch_accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('batch_accuracy', accuracy)

        summ = tf.summary.merge_all()

        with tf.Session() as session:
            session.run(init)
            self.writer.add_graph(session.graph)
            
            for epoch in range(num_epochs):
                epoch_cost = 0
                num_minibatches = int(num_examples / self.batch_size)
                for step in range(num_minibatches):
                    (minibatch_X, minibatch_Y) = self._next_batch(step)
                    minibatch_X = self._reformat(minibatch_X)
                    _, minibatch_cost, predictions = session.run([self.optimizer, self.cost, train_prediction], 
                            feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches
                    if step % 50 == 0:
                        print("Minibatch loss at step %d: %f" % (step, minibatch_cost))
                        print("Minibatch accuracy: %.1f%%" % batch_accuracy(predictions, minibatch_Y))

                    if step % 5 == 0:
                        [b, s] = session.run([accuracy, summ], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                        self.writer.add_summary(s, (epoch * num_minibatches) + step)

                if print_cost:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

                if print_cost == True and epoch % 1 == 0:
                    costs.append(epoch_cost)

            self._evaluate()

    def _create_placeholders(self):
        X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.num_channels], name='X')
        Y = tf.placeholder(tf.float32, [None, self.num_classes], name='Y')
        return X, Y

    def _forward_propagation(self, X, keep_prob=1.0, training=False):
        """Defines feed forward computation graph.
        
        Arguments:
          X: Input dataset placeholder, of shape (batch_size, image_size, 
            image_size, num_channels).
          parameters: Dictionary containing the initialized weights and biasses.
          keep_prob: Float, dropout probability hyperparameter.
          training: Boolean, used to apply dropout regularization during 
            training.

        Returns:
          fc2: The output of the last linear unit
        """
      
        conv1 = self._conv_layer(input=X, size_in=self.num_channels, 
                size_out=self.depth, patch_size=self.patch_size, conv_stride=1, 
                name='conv1')

        conv2 = self._conv_layer(input=conv1, size_in=self.depth, size_out=50, 
                patch_size=self.patch_size, conv_stride=2, name='conv2')
        
        shape = conv2.get_shape().as_list()
        flattened = tf.reshape(conv2, [-1, shape[1] * shape[2] * shape[3]]) 
        
        fc1 = self._fully_connected_layer(flattened, size_in=4*4*50, 
                size_out=self.dense_layer_units, name='fc1')

        fc1_dropout = self._dropout(fc1, keep_prob, training)
        
        fc2 = self._fully_connected_layer(fc1_dropout, 
                size_in=self.dense_layer_units, 
                size_out=self.num_classes, name='fc2')
        return fc2

    def _conv_layer(self, input, size_in, size_out, patch_size, conv_stride, name='conv'):
        with tf.variable_scope(name):
            W = tf.get_variable('W', [patch_size, patch_size, size_in, size_out], 
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [size_out], initializer=tf.zeros_initializer())

            conv = tf.nn.conv2d(input, W, strides=[1, conv_stride, conv_stride, 1], padding=self.padding)
            act = tf.nn.relu(conv + b)
            pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.padding)

            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            return pool

    def _fully_connected_layer(self, input, size_in, size_out, name='fc'):
        with tf.variable_scope(name):
            W = tf.get_variable('W', [size_in, size_out], 
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [size_out], 
                    initializer=tf.zeros_initializer())

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
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            tf.summary.scalar("cost", cost)
        return cost

    def _reformat(self, dataset):
        return dataset.reshape([-1, self.image_size, self.image_size, self.num_channels])
    
    def _evaluate(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print ("Train Accuracy:", accuracy.eval({self.X: self._reformat(self.X_train), self.Y: self.Y_train}))
            print ("Test Accuracy:", accuracy.eval({self.X: self._reformat(self.X_test), self.Y: self.Y_test}))

    def _next_batch(self, step):
        offset = (step * self.batch_size) % (self.X_train.shape[0] - self.batch_size)
        minibatch_X = self.X_train[offset:(offset + self.batch_size), :]
        minibatch_Y = self.Y_train[offset:(offset + self.batch_size), :]
        return minibatch_X, minibatch_Y


def main(_):
    dataset = load_dataset()
    X_train, Y_train = dataset.train.images, dataset.train.labels
    X_test, Y_test = dataset.test.images, dataset.test.labels

    fashion_classiffier = FashionClassifier(X_train, Y_train, X_test, Y_test, 
            image_size=28, num_channels=1, num_classes=10, log_dir='/tmp/fashion-classifier/6')

    fashion_classiffier.model(padding='SAME', patch_size=5, depth=20, 
            dense_layer_units=500, learning_rate=0.001, batch_size=128, 
            keep_prob=0.5)

    fashion_classiffier.train(num_epochs=10, print_cost=True)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[]) 
