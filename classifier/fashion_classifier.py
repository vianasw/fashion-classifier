#!/usr/bin/python3

from utils.dataset import load_dataset, shuffle_dataset
from utils.data_augmentation import augment_data
from utils.visualization import create_sprite_image, invert_grayscale
from models.lenet import LeNet
from models.vgg import VGG
from tensorflow.contrib.tensorboard.plugins import projector
from collections import namedtuple
import argparse
import tensorflow as tf
import numpy as np
import os
import sys

FLAGS = None
DEFAULT_LOGDIR = '/tmp/fashion-classifier/logdir/'

models = {'lenet': LeNet, 'vgg': VGG}

# Tuple containing numpy arrays for training examples and labels
DatasetPair = namedtuple('DatasetPair', ['X', 'Y'])

class FashionClassifier:
    """Classifier for the Fashion-MNIST Zalando's dataset [1].

    Convolutional neural network to classify Zalando's article images from 10
    different classes (i.e. trouser, dress, coat, bag, etc).
    [1]: https://github.com/zalandoresearch/fashion-mnist


    Arguments:
        model: Model, initialized model to train
        train_dataset: DatasetPair of training examples of shape [num_examples,
            image_size * image_size] and labels of shape [num_examples, 1]
        test_dataset: DatasetPair of test examples of shape [num_examples,
            image_size * image_size] and labels of shape [num_examples, 1]
        batch_size: Integer, size of the batch to process in each training step
        log_dir: Path to save logs for Tensorboard and checkpoint files
        checkpoint_filename: name of the checkpoint file, default: model.ckpt
    """
    def __init__(self, model, train_dataset, test_dataset, batch_size,
                 log_dir, checkpoint_filename='model.ckpt'):
        self.model = model
        self.X_train = train_dataset.X
        self.Y_train = train_dataset.Y
        self.X_test = test_dataset.X
        self.Y_test = test_dataset.Y
        self.writer = tf.summary.FileWriter(log_dir)
        self.log_dir = log_dir
        self.checkpoint_filename = checkpoint_filename

        self.batch_size = batch_size

        self.embedding_placeholder = tf.placeholder(
            tf.float32, shape=[None, self.model.image_size * self.model.image_size],
            name="embedding_x")

        # this is to display some image examples on Tensorboard
        tf.summary.image('input', self.model.X(), 3)

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.current_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='current_step')

    def train_and_evaluate(self, num_epochs, resume_training=False,
                           print_cost=False, create_embeddings=False):
        """Performs training for the CNN defined with the model method.

        Arguments:
            num_epochs: Integer, number of epochs to perform training.
            resume_training: Boolean, if True restores tensor values from
            checkpoint
            print_cost: Boolean, if True, prints costs every some iterations
            create_embeddings: Boolean, if True will create embeddings visualization
        """
        logits = self.model.logits()
        accuracy = self.model.accuracy(logits)

        num_examples = self.X_train.shape[0]
        shuffled_X, shuffled_Y = shuffle_dataset(self.X_train, self.Y_train)

        summ_op = tf.summary.merge_all()

        embedding_assign = None
        if create_embeddings:
            embedding_assign = self._create_embeddings()

        optimizer = self.model.optimizer(self.global_step)
        tf_metric, tf_metric_update = tf.metrics.accuracy(labels=tf.argmax(self.model.Y(), 1), predictions=tf.argmax(logits, 1))
        init = tf.global_variables_initializer()

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
                    (minibatch_X, minibatch_Y) = self._next_batch(step, self.X_train, self.Y_train, self.batch_size)

                    _, minibatch_cost = session.run(
                        [optimizer, self.model.cost],
                        feed_dict=self.model.feed_dict(minibatch_X, minibatch_Y))

                    epoch_cost += minibatch_cost / num_minibatches

                    self._log_progress(
                        session, summ_op, accuracy, minibatch_cost,
                        num_minibatches, minibatch_X, minibatch_Y, epoch,
                        step, print_cost)

                    self._save_checkpoint(session, saver, step,
                                          embedding_assign)
                current_step = 0

                # Handling the end case (last mini-batch < batch_size)
                if num_examples % num_minibatches != 0:
                    (minibatch_X, minibatch_Y) = self._last_batch(self.X_train, self.Y_train, self.batch_size, num_minibatches)
                    _, minibatch_cost = session.run(
                        [optimizer, self.model.cost],
                        feed_dict=self.model.feed_dict(minibatch_X, minibatch_Y))

                    epoch_cost += minibatch_cost / num_minibatches

                if print_cost:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))

            self._print_accuracy(session, logits, tf_metric, tf_metric_update)

    def load_and_evaluate(self):
        """Loads model from last checkpoint stored in log_dir."""
        eval_logits = self.model.logits()

        tf_metric, tf_metric_update = tf.metrics.accuracy(labels=tf.argmax(self.model.Y(), 1), predictions=tf.argmax(eval_logits, 1))
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            self._restore_last_checkpoint(session, saver)
            self._print_accuracy(session, eval_logits, tf_metric, tf_metric_update)

    def _reformat(self, dataset):
        return dataset.reshape([-1, self.model.image_size, self.model.image_size,
                               self.model.num_channels])

    def _print_accuracy(self, session, logits, tf_metric, tf_metric_update):
        train_accuracy = self._compute_accuracy(session, logits, tf_metric, tf_metric_update,
                self.X_train, self.Y_train)
        print("Train accuracy: ", train_accuracy)
        test_accuracy = self._compute_accuracy(session, logits, tf_metric, tf_metric_update,
                self.X_test, self.Y_test)
        print("Test accuracy: ", test_accuracy)

    def _compute_accuracy(self, session, logits, tf_metric, tf_metric_update, X, Y):
        num_examples = self.X_train.shape[0]
        batch_size = min(1000, num_examples - 1)
        num_minibatches = int(num_examples / batch_size)
        session.run(tf.local_variables_initializer())
        for step in range(num_minibatches):
            (minibatch_X, minibatch_Y) = self._next_batch(step, X, Y, batch_size)
            tf_metric_update.eval(self.model.feed_dict(minibatch_X, minibatch_Y))

        if num_examples % num_minibatches != 0:
            (minibatch_X, minibatch_Y) = self._last_batch(X, Y, batch_size, num_minibatches)
            tf_metric_update.eval(self.model.feed_dict(minibatch_X, minibatch_Y))

        accuracy = tf_metric.eval()
        return accuracy

    def _next_batch(self, step, X, Y, batch_size):
        offset = (step * batch_size) % (X.shape[0] - batch_size)
        minibatch_X = X[offset:(offset + batch_size), :]
        minibatch_Y = Y[offset:(offset + batch_size), :]
        minibatch_X = self._reformat(minibatch_X)
        return minibatch_X, minibatch_Y

    def _last_batch(self, X, Y, batch_size, num_minibatches):
        offset = num_minibatches * batch_size
        minibatch_X = X[offset:, :]
        minibatch_Y = Y[offset:, :]
        minibatch_X = self._reformat(minibatch_X)
        return minibatch_X, minibatch_Y

    def _create_embeddings(self):
        metadata_path = os.path.join(self.log_dir, 'metadata.tsv')
        image_path = os.path.join(self.log_dir, 'sprite.png')
        self._create_labels_if_not_exists(metadata_path)
        self._create_sprite_if_not_exists(image_path)

        embedding = tf.Variable(
            tf.zeros([self.dense_layer_units, self.dense_layer_units]),
            name='embedding')

        emb_assign = embedding.assign(self.embedding_input)
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        embedding_config.metadata_path = metadata_path
        embedding_config.sprite.image_path = image_path
        embedding_config.sprite.single_image_dim.extend([self.image_size, self.image_size])
        projector.visualize_embeddings(self.writer, config)
        return emb_assign

    def _create_sprite_if_not_exists(self, save_path):
        if not os.path.exists(save_path):
            X_embedding = self.X_test[:self.dense_layer_units]
            embedding_imgs = X_embedding.reshape(
                [-1, self.image_size, self.image_size])
            embedding_imgs = invert_grayscale(embedding_imgs)
            create_sprite_image(embedding_imgs, save_path)

    def _create_labels_if_not_exists(self, save_path):
        if not os.path.exists(save_path):
            Y_embedding = self.Y_test[:self.dense_layer_units]
            with open(save_path, 'w') as f:
                f.write("Index\tLabel\n")
                for index, one_hot_label in enumerate(Y_embedding):
                    label = np.argmax(one_hot_label)
                    f.write("%d\t%d\n" % (index, label))

    def _log_progress(self, session, summ, accuracy, minibatch_cost,
                      num_minibatches, minibatch_X, minibatch_Y, epoch, step,
                      print_cost):
        if step % 5 == 0:
            batch_accuracy, s = session.run([accuracy, summ], 
                    feed_dict=self.model.feed_dict(minibatch_X, minibatch_Y))
            self.writer.add_summary(s, (epoch * num_minibatches) + step)
            if print_cost and step % 50 == 0:
                print("Minibatch loss at step %d: %f" % (step, minibatch_cost))
                print("Minibatch accuracy: %.1f%%" % (batch_accuracy * 100))

    def _save_checkpoint(self, session, saver, step, embedding_assign):
        session.run(tf.assign(self.current_step, step))
        if step % 200 == 0:
            if embedding_assign is not None:
                batch_x = self._reformat(self.X_test[:self.dense_layer_units])
                session.run(embedding_assign,
                            feed_dict={self.X: batch_x,
                                       self.Y: self.Y_test[:self.dense_layer_units]})

            checkpoint_path = os.path.join(self.log_dir,
                                           self.checkpoint_filename)

            saver.save(session, checkpoint_path, global_step=self.global_step)

    def _restore_last_checkpoint(self, session, saver):
        assert tf.gfile.Exists(os.path.join(self.log_dir, 'checkpoint'))
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        saver.restore(session, ckpt.model_checkpoint_path)


def main(_):
    dataset = load_dataset()
    train_dataset = DatasetPair(dataset.train.images, dataset.train.labels)
    test_dataset = DatasetPair(dataset.test.images, dataset.test.labels)

    log_dir = FLAGS.logdir

    hparams = get_hparams(FLAGS.hparams)
    # Data augmentation: apply random horizontal flip and random crop
    if hparams.augment_percent > 0:
        images, labels = augment_data(train_dataset.X, train_dataset.Y, 28, 28, 1, hparams.augment_percent)
        train_dataset = DatasetPair(images, labels)

    model_class = models[FLAGS.model]
    model = model_class(hparams, image_size=28, num_channels=1, num_classes=10)
    fashion_classiffier = FashionClassifier(model, train_dataset, test_dataset,
                                            batch_size=hparams.batch_size, log_dir=log_dir)
    if FLAGS.action == 'train':
        resume_training = FLAGS.resume_training
        create_embeddings = FLAGS.create_embeddings
        fashion_classiffier.train_and_evaluate(num_epochs=hparams.num_epochs,
                                               resume_training=resume_training,
                                               print_cost=True,
                                               create_embeddings=create_embeddings)
    elif FLAGS.action == 'load':
        fashion_classiffier.load_and_evaluate()

def get_hparams(hparams_str):
    """Parses hparams_str to HParams object.

    Arguments:
        hparams_str: String of comma separated param=value pairs.

    Returns:
        hparams: tf.contrib.training.HParams object from hparams_str. If
            hparams_str is None, then a default HParams object is returned.
    """
    hparams = tf.contrib.training.HParams(learning_rate=0.001, conv1_depth=32,
                                          conv2_depth=128, dense_layer_units=1024,
                                          batch_size=128, padding='SAME',
                                          keep_prob=0.5, lambd=0.0,
                                          num_epochs=1, patch_size=5,
                                          augment_percent=0.0)
    if hparams_str:
        hparams.parse(hparams_str)
    return hparams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('action', choices=['train', 'load'], default=None)
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOGDIR,
                        help='Store log/model files.')
    parser.add_argument('--resume_training', action='store_true',
                        default=False,
                        help='Resume training by loading last checkpoint')
    parser.add_argument('--hparams', type=str, default=None,
                        help='Comma separated list of "name=value" pairs.')
    parser.add_argument('--create_embeddings', action='store_true',
                        default=False,
                        help='Create embeddings during training.')
    parser.add_argument('--model', default='vgg', const='vgg', nargs='?',
                        choices=['lenet', 'vgg'],
                        help='Model to use during training.')

    FLAGS, _ = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]])
