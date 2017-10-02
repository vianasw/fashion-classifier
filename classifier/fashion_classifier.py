#!/usr/bin/python3

from utils import load_dataset, plot_costs, accuracy
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
      dataset: DataSets object as returned by tensorflow input_data.read_data_sets which contains
        train and test datasets.
      image_size: Integer, specifies the width and height of the images from 
        the datasets.
      num_channels: Integer, number of channels of the image (1 for greyscale, 
        3 for RGB).
      num_classes: Integer, number of classes to predict.
    """
    def __init__(self, dataset, image_size, num_channels, num_classes):
        self.dataset = dataset
        self.parameters = dict()
        self.padding = 'SAME'
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.X = None
        self.Y = None
        self.cost = None
        self.optimizer = None
        self.logits = None
        self.batch_size = 64
        self.learning_rate = 0.001

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
        self.X, self.Y = self._create_placeholders()
        test_dataset = self._reformat(self.dataset.test.images)
        self.X_test = tf.constant(test_dataset)

        self.parameters = self._initialize_parameters(patch_size, depth, dense_layer_units)
        self.logits = self._forward_propagation(self.X, self.parameters, keep_prob, training=True)
        self.cost = self._compute_cost(self.logits, self.Y)
        self.learning_rate = learning_rate

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
        test_prediction = tf.nn.softmax(self._forward_propagation(self.X_test, self.parameters))

        with tf.Session() as session:
            session.run(init)
            
            num_examples = self.dataset.train.num_examples
            for epoch in range(num_epochs):
                epoch_cost = 0
                num_minibatches = int(num_examples / self.batch_size)
                for i in range(num_minibatches):
                    (minibatch_X, minibatch_Y) = self.dataset.train.next_batch(self.batch_size)
                    minibatch_X = self._reformat(minibatch_X)
                    _, minibatch_cost, predictions = session.run([self.optimizer, self.cost, train_prediction], 
                            feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches
                    if i % 50 == 0:
                        print("Minibatch loss at step %d: %f" % (i, minibatch_cost))
                        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, minibatch_Y))

                if print_cost:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

                if print_cost == True and epoch % 1 == 0:
                    costs.append(epoch_cost)

            if print_cost:
                plot_costs(costs, title="Learning rate =" + str(self.learning_rate))

            self.parameters = session.run(self.parameters)

            self._evaluate(test_prediction)

    def _create_placeholders(self):
        X = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.num_channels], name='X')
        Y = tf.placeholder(tf.float32, [self.batch_size, self.num_classes], name='Y')
        return X, Y

    def _initialize_parameters(self, patch_size, depth, dense_layer_units):
        """ Initializes weights and biases parameters.
        
        Weights are initialized with Xavier initialization, biases with zeros.

        Arguments:
          patch_size: Integer, convolution patch size.
          depth: Integer, depth of first conv layer.
          dense_layer_units: Integer, number of hidden units in the dense 
            layer.

        Returns:
          parameters: Dictionary containing initialized weights and biases.
        """
        W1 = tf.get_variable('W1', [patch_size, patch_size, self.num_channels, depth], 
            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1', [depth], initializer=tf.zeros_initializer())

        W2 = tf.get_variable('W2', [patch_size, patch_size, depth, 50], 
            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', [50], initializer=tf.zeros_initializer())

        W3 = tf.get_variable('W3', [4 * 4 * 50, dense_layer_units], 
            initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('b3', [dense_layer_units], initializer=tf.zeros_initializer())

        W4 = tf.get_variable('W4', [dense_layer_units, self.num_classes], 
            initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable('b4', [self.num_classes], initializer=tf.zeros_initializer())
        
        parameters = dict()
        parameters['W1'] = W1
        parameters['b1'] = b1
        parameters['W2'] = W2
        parameters['b2'] = b2
        parameters['W3'] = W3
        parameters['b3'] = b3
        parameters['W4'] = W4
        parameters['b4'] = b4

        return parameters

    def _forward_propagation(self, X, parameters, keep_prob=1.0, training=False):
        """Defines feed forward computation graph.
        
        Arguments:
          X: Input dataset placeholder, of shape (batch_size, image_size, 
            image_size, num_channels).
          parameters: Dictionary containing the initialized weights and biasses.
          keep_prob: Float, dropout probability hyperparameter.
          training: Boolean, used to apply dropout regularization during 
            training.

        Returns:
          Z4: The output of the last linear unit
        """

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']

        Z1 = tf.nn.conv2d(X, W1, [1, 1, 1, 1], padding=self.padding) + b1
        A1 = tf.nn.relu(Z1)
        M1 = tf.nn.max_pool(A1, [1, 2, 2, 1], [1, 2, 2, 1], padding=self.padding)

        Z2 = tf.nn.conv2d(M1, W2, [1, 2, 2, 1], padding=self.padding) + b2
        A2 = tf.nn.relu(Z2)
        M2 = tf.nn.max_pool(A2, [1, 2, 2, 1], [1, 2, 2, 1], padding=self.padding)

        shape = M2.get_shape().as_list()
        M2_flatten = tf.reshape(M2, [shape[0], shape[1] * shape[2] * shape[3]])
        Z3 = tf.matmul(M2_flatten, W3) + b3
        A3 = tf.nn.relu(Z3)

        if training:
            keep_prob = tf.constant(keep_prob)
        else:
            keep_prob = tf.constant(1.0)

        A3_dropout = tf.nn.dropout(A3, keep_prob)

        Z4 = tf.matmul(A3_dropout, W4) + b4
        return Z4

    def _compute_cost(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def _reformat(self, dataset):
        return dataset.reshape([-1, self.image_size, self.image_size, self.num_channels])
    
    def _evaluate(self, test_prediction):
        print ("Test Accuracy: %.1f%%" % accuracy(test_prediction.eval(), self.dataset.test.labels))

def main(_):
    dataset = load_dataset()
    fashion_classiffier = FashionClassifier(dataset, image_size=28, num_channels=1, num_classes=10)
    fashion_classiffier.model(padding='SAME', patch_size=5, depth=20, dense_layer_units=500, 
            learning_rate=0.001, batch_size=128, keep_prob=0.5)
    fashion_classiffier.train(num_epochs=5, print_cost=True)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[]) 
