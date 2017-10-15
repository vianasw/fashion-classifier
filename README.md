# Fashion-Classifier

`Fashion-Classifier` is a proof-of-concept classifier for [Fashion-MNIST Zalando's dataset](https://github.com/zalandoresearch/fashion-mnist), to test some deep learning techiniques as well as some technologies such as Tensorflow and Tensorboard. I'm using Zalando's dataset instead of the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) because there are already so many MNIST Tensorflow tutorials out there and because it was a lot more challenging than MNIST.

## Architecture
I decided to start with a slightly modified [LeNet-5](http://yann.lecun.com/exdb/lenet/) architecture. LeNet-5 is a very simple and well known convolutional neural network architecture ([LeNet-5](http://yann.lecun.com/exdb/lenet/)) that is easy to implement and usually gives good results out of the box to start with. 

I've done three main modifications to the original LeNet-5: 

* I'm using _ReLU_ as activation function instead of _tanh_ 
* The depths of the convolutional layers.
* The humber of hidden units of the fully connected layer.

![](doc/img/cnn-architecture.png)
<img src="doc/img/cnn-architecture.png">

## Hyperparameters search

To fine tune the model and explore the hyperparameters space, I've implemented a few helper functions in `hparams_search.py` to perform random grid search following a coarse to fine approach.

`hparams_search.py` usage:
* --logdir path to dir where log/model files will be stored.
* --hparams_path path to json file that specifies the values of the hyperparameters to train.
* --grid_size indicates the number of hyperparameter combinations to test from all the possible combinations.

```
$ python hparams_search.py --logdir /tmp/fashion-classifier/coarse/ --hparams_path coarse.json --grid_size 10
```

The grid of hyperparameters to explore are specified in a json file with the name of each hyperparameter and a list of possible values. For example:

```
{
    "conv1_depth": [16, 32, 64],
    "conv2_depth": [32, 64, 128],
    "dense_layer_units": [512, 1024, 2048],
    "batch_size": [64, 128, 256, 512],
    "keep_prob": [0.4, 0.5, 0.6],
    "num_epochs": [5]
}
```

A cartesian product is applied to every hyperparameter list of values to generate all possible combinations. The list of all possible combinations is randomized and the `--grid_size` argument limits the number of combinations to try.

## Data augmentation
To reduce overfitting I've implemented two methods to synthetically generate more examples:

* Random horizontal flip.
* Random crop with resize.

However, the LeNet model architecture was probably too shallow to absorb those new examples and it didn't help much to reduce overfitting.

## Visualizations

## Results

## Conclusion