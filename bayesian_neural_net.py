# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a Bayesian neural network to classify MNIST digits.
The architecture is LeNet-5 [1].
#### References
[1]: Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
     Gradient-based learning applied to document recognition.
     _Proceedings of the IEEE_, 1998.
     http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import flags
import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from tensorflow.contrib.learn.python.learn.datasets import mnist



# TODO(b/78137893): Integration tests currently fail with seaborn imports.
warnings.simplefilter(action="ignore")

try:
    import seaborn as sns  # pylint: disable=g-import-not-at-top
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

tfd = tfp.distributions

ISING = True

IMAGE_SHAPE = [40,40,1] if ISING else [28, 28, 1] 

flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=6000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "bayesian_neural_network/data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         "bayesian_neural_network/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=400,
                     help="Frequency at which save visualizations.")
flags.DEFINE_integer("num_monte_carlo",
                     default=10,
                     help="Network draws to compute predictive probabilities.")
flags.DEFINE_bool("fake_data",
                  default=None,
                  help="If true, uses fake data. Defaults to real data.")

FLAGS = flags.FLAGS

def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    """Save a PNG plot with histograms of weight means and stddevs.
    Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
        whose elements are Numpy `array`s, of any shape, containing
        posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
        whose elements are Numpy `array`s, of any shape, containing
        posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
    """
    fig = figure.Figure(figsize=(6, 3))
    canvas = backend_agg.FigureCanvasAgg(fig)

    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.distplot(qm.flatten(), ax=ax, label=n)
    ax.set_title("weight means")
    ax.set_xlim([-1.5, 1.5])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.distplot(qs.flatten(), ax=ax)
    ax.set_title("weight stddevs")
    ax.set_xlim([0, 1.])

    fig.tight_layout()
    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))


def plot_heldout_prediction(input_vals, label_vals, probs,
                            fname, n=10, title=""):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.
    Args:
    input_vals: A `float`-like Numpy `array` of shape
        `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
        num_heldout, num_classes]` containing Monte Carlo samples of
        class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
    """
    fig = figure.Figure(figsize=(9, 3*n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    indices = np.random.randint(low=0,high=input_vals.shape[0],size=n)
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3*i + 1)
        ax.imshow(input_vals[indices[i], :].reshape(IMAGE_SHAPE[:-1]), interpolation="None")

        ax = fig.add_subplot(n, 3, 3*i + 2)
        for prob_sample in probs:
            sns.barplot(np.arange(2) if ISING else np.arange(10), prob_sample[indices[i], :], alpha=0.5 if ISING else 0.1, ax=ax)
            ax.set_ylim([0, 1])
        ax.set_title("posterior samples")

        ax = fig.add_subplot(n, 3, 3*i + 3)
        sns.barplot(np.arange(2) if ISING else np.arange(10), np.mean(probs[:, indices[i], :], axis=0), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("predictive probs, correct=%.i" % label_vals[indices[i]] )
        
    fig.suptitle(title)
    fig.tight_layout()

    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))



def plot_test_prediction(input_vals, probs,
                            fname, n=10, title=""):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.
    Args:
    input_vals: A `float`-like Numpy `array` of shape
        `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
        num_heldout, num_classes]` containing Monte Carlo samples of
        class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
    """
    fig = figure.Figure(figsize=(9, 3*n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    indices = np.random.randint(low=0,high=input_vals.shape[0],size=n)
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3*i + 1)
        ax.imshow(input_vals[indices[i], :].reshape(IMAGE_SHAPE[:-1]), interpolation="None")

        ax = fig.add_subplot(n, 3, 3*i + 2)
        for prob_sample in probs:
            sns.barplot(np.arange(2) if ISING else np.arange(10), prob_sample[indices[i], :], alpha=0.5 if ISING else 0.1, ax=ax)
            ax.set_ylim([0, 1])
        ax.set_title("posterior samples")

        ax = fig.add_subplot(n, 3, 3*i + 3)
        sns.barplot(np.arange(2) if ISING else np.arange(10), np.mean(probs[:, indices[i], :], axis=0), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title("predictive probs, test set")
        
    fig.suptitle(title)
    fig.tight_layout()

    canvas.print_figure(fname, format="png")
    print("saved {}".format(fname))

def build_input_pipeline(mnist_data, batch_size, heldout_size):
    """Build an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.train.images, np.int32(mnist_data.train.labels)))

    print(mnist_data.train.images.shape)
    training_batches = training_dataset.shuffle(
        50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)

    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.validation.images,
        np.int32(mnist_data.validation.labels)))
    heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
    heldout_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_frozen)


    test_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.test.images,
        np.int32(mnist_data.test.labels)))
    test_frozen = (test_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
    test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_frozen)


    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator, test_iterator


def test_data_pipeline(mnist_data, batch_size):
    """Build an Iterator switching between train and heldout data."""


    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices(
        (mnist_data.test.images))
    heldout_frozen = (heldout_dataset.take(batch_size).
                    repeat().batch(batch_size))
    test_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_frozen)

    # Combine these into a feedable iterator that can switch between training
    # and test inputs.
    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle, heldout_dataset.output_types, heldout_dataset.output_shapes)
    images = feedable_iterator.get_next()

    return images, handle, test_iterator






def Get_ising_data():
    import pickle
    
    def read_t(t,root="/home/samknu/MyRepos/MLProjectIsingModel/data/IsingData/"):
        data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
        return np.unpackbits(data).astype(int).reshape(-1,1600)
    
    temperatures = np.arange(0.25, 4., step=0.25)
    
    ordered = np.zeros(shape=(np.sum(temperatures<2.0),10000,1600))
    disordered = np.zeros(shape=(np.sum(temperatures>2.5),10000,1600))
    critical = np.zeros(shape=(np.sum((temperatures>=2.0)*(temperatures<=2.5)),10000,1600))
    
    ordered_index = 0
    disordered_index = 0
    crit_index = 0
    for i in range(len(temperatures)):
        T = temperatures[i]
        if T < 2.0:
            ordered[ordered_index] = read_t(T)
            ordered_index += 1
        elif T > 2.5:
            disordered[disordered_index] = read_t(T)
            disordered_index += 1
        else:
            critical[crit_index] = read_t(T)
            crit_index += 1

    ordered = ordered.reshape(-1,1600)       # 70000
    disordered = disordered.reshape(-1,1600) # 50000
    critical = critical.reshape(-1,1600)     # 30000

    # Shuffling before separating into training, validation and test set
    np.random.shuffle(ordered)
    np.random.shuffle(disordered)
    np.random.shuffle(critical)

    training_data = np.zeros((6000*12,1600))
    validation_data = np.zeros((2000*12,1600))
    test_data = np.zeros((2000*12 + 10000*3,1600))

    training_data[:round(0.6*70000)] = ordered[:round(0.6*70000)]
    training_data[round(0.6*70000):] = disordered[:round(0.6*50000)]

    validation_data[:round(0.2*70000)] = ordered[round(0.6*70000):round(0.6*70000)+round(0.2*70000)]
    validation_data[round(0.2*70000):] = disordered[round(0.6*50000):round(0.6*50000)+round(0.2*50000)]

    test_data[:round(0.2*70000)] = ordered[round(0.6*70000)+round(0.2*70000):round(0.6*70000)+2*round(0.2*70000)]
    test_data[round(0.2*70000):round(0.2*70000)+round(0.2*50000)] = disordered[round(0.6*50000)+round(0.2*50000):round(0.6*50000)+2*round(0.2*50000)]
    test_data[round(0.2*70000)+round(0.2*50000):] = critical

    training_labels = np.zeros(6000*12)
    training_labels[round(0.6*70000):] = np.ones(round(0.6*50000))

    validation_labels = np.zeros(2000*12)
    validation_labels[round(0.2*70000):] = np.ones(round(0.2*50000))

    # Class 0 is ordered, class 1 is disordered

    ############################################################
    # Reshaping since we want them as matrices for convolution #
    ############################################################
    training_data = training_data.reshape(-1,40,40)
    training_data = training_data[:,:,:,np.newaxis]

    validation_data = validation_data.reshape(-1,40,40)
    validation_data = validation_data[:,:,:,np.newaxis]
    
    test_data = test_data.reshape(-1,40,40)
    test_data = test_data[:,:,:,np.newaxis]
    

    del ordered
    del disordered
    del critical
    del temperatures

    
    #############################
    # Shuffling data and labels #
    #############################
    indices = np.random.permutation(np.arange(training_data.shape[0]))
    training_data = training_data[indices]
    training_labels = training_labels[indices]
    
    indices = np.random.permutation(np.arange(validation_data.shape[0]))
    validation_data = validation_data[indices]
    validation_labels = validation_labels[indices]
    
    indices = np.random.permutation(np.arange(test_data.shape[0]))
    test_data = test_data[indices]
    #test_labels = test_labels[indices]
    
    cut_train = 20000   
    cut_val = 5000
    cut_test = 100
    training_data = training_data[:cut_train]
    training_labels = training_labels[:cut_train]

    validation_data = validation_data[:cut_val]
    validation_labels = validation_labels[:cut_val]
    
    test_data = test_data[:cut_test]

    class Dummy(object):
        pass
    ising_data = Dummy()
    ising_data.train=Dummy()
    ising_data.train.images = training_data
    ising_data.train.labels = training_labels
    ising_data.train.num_examples = training_data.shape[0]

    ising_data.validation=Dummy()
    ising_data.validation.images = validation_data
    ising_data.validation.labels = validation_labels
    ising_data.validation.num_examples = validation_data.shape[0]


    ising_data.test=Dummy()
    ising_data.test.images = test_data
    ising_data.test.labels = np.zeros(test_data.shape[0])   # dummy labels
    ising_data.test.num_examples = test_data.shape[0]

    return ising_data






def main(argv):
    del argv  # unused

    if tf.io.gfile.exists(FLAGS.model_dir):
        tf.compat.v1.logging.warning(
            "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
        tf.io.gfile.rmtree(FLAGS.model_dir)
    tf.io.gfile.makedirs(FLAGS.model_dir)



    if ISING:
        the_data = Get_ising_data()
    else:
        the_data = mnist.read_data_sets(FLAGS.data_dir, reshape=False)

    
    (images, labels, handle, training_iterator, heldout_iterator, test_iterator) = build_input_pipeline(
           the_data, FLAGS.batch_size, the_data.validation.num_examples)  


    # Build a Bayesian LeNet5 network. We use the Flipout Monte Carlo estimator
    # for the convolution and fully-connected layers: this enables lower
    # variance stochastic gradients than naive reparameterization.
    with tf.compat.v1.name_scope("bayesian_neural_net", values=[images]):
        neural_net = tf.keras.Sequential([
            tfp.layers.Convolution2DFlipout(6,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                            strides=[2, 2],
                                            padding="SAME"),
            tfp.layers.Convolution2DFlipout(16,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                            strides=[2, 2],
                                            padding="SAME"),
            tfp.layers.Convolution2DFlipout(120,
                                            kernel_size=5,
                                            padding="SAME",
                                            activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseFlipout(84, activation=tf.nn.relu),
            tfp.layers.DenseFlipout(2) if ISING else tfp.layers.DenseFlipout(10)
            ])
    
    logits = neural_net(images)
    labels_distribution = tfd.Categorical(logits=logits)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(
        input_tensor=labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / the_data.train.num_examples     # 72000 is the size of the training set
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for validation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(input=logits, axis=1)
    accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=predictions)

    # Extract weight posterior statistics for layers with weight distributions
    # for later visualization.
    names = []
    qmeans = []
    qstds = []
    for i, layer in enumerate(neural_net.layers):
        try:
            q = layer.kernel_posterior
        except AttributeError:
            continue
        names.append("Layer {}".format(i))
        qmeans.append(q.mean())
        qstds.append(q.stddev())


    with tf.compat.v1.name_scope("train"):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(elbo_loss)

    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer())

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        # Run the training loop.
        train_handle = sess.run(training_iterator.string_handle())
        heldout_handle = sess.run(heldout_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        
        for step in range(FLAGS.max_steps):
            #for step in range(0):
            _ = sess.run([train_op, accuracy_update_op],
                        feed_dict={handle: train_handle})
            if step % 100 == 0:
                loss_value, accuracy_value = sess.run(
                    [elbo_loss, accuracy], feed_dict={handle: train_handle})
                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
                    step, loss_value, accuracy_value))

            if (step+1) % FLAGS.viz_steps == 0:
                # Compute log prob of heldout set by averaging draws from the model:
                # p(heldout | train) = int_model p(heldout|model) p(model|train)
                #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
                # where model_i is a draw from the posterior p(model|train).
                probs = np.asarray([sess.run((labels_distribution.probs),
                                            feed_dict={handle: heldout_handle})
                                    for _ in range(FLAGS.num_monte_carlo)])
                mean_probs = np.mean(probs, axis=0)

                image_vals, label_vals = sess.run((images, labels),
                                                feed_dict={handle: heldout_handle})
                image_vals_test = sess.run((images),
                                                feed_dict={handle: test_handle})

                heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                                    label_vals.flatten()]))
                                                    
                print(" ... Held-out nats: {:.3f}".format(heldout_lp))

                qm_vals, qs_vals = sess.run((qmeans, qstds))

                if HAS_SEABORN:
                    plot_weight_posteriors(names, qm_vals, qs_vals,
                                            fname=os.path.join(
                                                FLAGS.model_dir,
                                                "step{:05d}_weights.png".format(step)))

                    plot_heldout_prediction(image_vals, label_vals, probs,
                                            fname=os.path.join(
                                                FLAGS.model_dir,
                                                "step{:05d}_pred.png".format(step)),
                                            title="mean heldout logprob {:.2f}"
                                            .format(heldout_lp))

                    plot_test_prediction(image_vals_test, probs,
                                            fname=os.path.join(
                                                FLAGS.model_dir,
                                                "step{:05d}_test_pred.png".format(step)))


if __name__ == "__main__":
    
    tf.compat.v1.app.run()      # this thing will run the main(argv) function with sys.argv as argument