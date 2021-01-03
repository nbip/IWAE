import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow_probability import distributions as tfd
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt


def logmeanexp(log_w, axis):
    max = tf.reduce_max(log_w, axis=axis)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis)) + max


def get_bias():
    # ---- For initializing the bias in the final Bernoulli layer for p(x|z)
    (Xtrain, ytrain), (_, _) = keras.datasets.mnist.load_data()
    Ntrain = Xtrain.shape[0]

    # ---- reshape to vectors
    Xtrain = Xtrain.reshape(Ntrain, -1) / 255

    train_mean = np.mean(Xtrain, axis=0)

    bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)

    return tf.constant_initializer(bias)


# ---- dynamically binarize the data
def bernoullisample(x):
    return np.random.binomial(1, x, size=x.shape).astype('float32')


def static_binarization_mnist(fashion=False):

    # ---- load data
    if fashion:
        (Xtrain, ytrain), (Xtest, ytest) = keras.datasets.fashion_mnist.load_data()
    else:
        (Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()
    Ntrain = Xtrain.shape[0]
    Ntest = Xtest.shape[0]

    # ---- reshape to vectors
    Xtrain = Xtrain.reshape(Ntrain, -1)
    Xtest = Xtest.reshape(Ntest, -1) / 255

    # ---- create validation set
    Xval = Xtrain[-10000:] / 255
    Xtrain = Xtrain[:-10000] / 255
    yval = ytrain[-10000:]
    ytrain = ytrain[:-10000]

    Ntrain = Xtrain.shape[0]
    Ntest = Xtest.shape[0]
    Nval = Xval.shape[0]

    # ---- dynamic binarize the data
    def bernoullisample(x):
        return np.random.binomial(1, x, size=x.shape).astype('float32')

    Xtrain = bernoullisample(Xtrain)
    Xval = bernoullisample(Xval)
    Xtest = bernoullisample(Xtest)

    p = np.random.permutation(Ntrain)
    Xtrain = Xtrain[p, :]
    ytrain = ytrain[p]

    p = np.random.permutation(Nval)
    Xval = Xval[p, :]
    yval = yval[p]

    p = np.random.permutation(Ntest)
    Xtest = Xtest[p, :]
    ytest = ytest[p]

    return Xtrain, Xval, Xtest, ytrain, yval, ytest


def dynamic_binarization_mnist():
    # ---- load data
    (Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()

    # ---- create validation set
    Xval = Xtrain[-10000:]
    Xtrain = Xtrain[:-10000]
    yval = ytrain[-10000:]
    ytrain = ytrain[:-10000]

    Ntrain = Xtrain.shape[0]
    Nval = Xval.shape[0]
    Ntest = Xtest.shape[0]

    # ---- reshape to vectors
    Xtrain = Xtrain.reshape(Ntrain, -1) / 255
    Xval = Xval.reshape(Nval, -1) / 255
    Xtest = Xtest.reshape(Ntest, -1) / 255

    return (Xtrain, ytrain), (Xval, yval), (Xtest, ytest)


def mnist(fashion=False):

    # ---- load data
    if fashion:
        (Xtrain, ytrain), (Xtest, ytest) = keras.datasets.fashion_mnist.load_data()
    else:
        (Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()
    Ntrain = Xtrain.shape[0]
    Ntest = Xtest.shape[0]

    # ---- reshape to vectors
    Xtrain = Xtrain.reshape(Ntrain, -1)
    Xtest = Xtest.reshape(Ntest, -1) / 255

    # ---- create validation set
    Xval = Xtrain[-10000:] / 255
    Xtrain = Xtrain[:-10000] / 255
    yval = ytrain[-10000:]
    ytrain = ytrain[:-10000]

    Ntrain = Xtrain.shape[0]
    Ntest = Xtest.shape[0]
    Nval = Xval.shape[0]

    # ---- dynamic binarize the data
    def bernoullisample(x):
        return np.random.binomial(1, x, size=x.shape).astype('float32')

    Xtrain = bernoullisample(Xtrain)
    Xval = bernoullisample(Xval)
    Xtest = bernoullisample(Xtest)

    p = np.random.permutation(Ntrain)
    Xtrain = Xtrain[p, :]
    ytrain = ytrain[p]

    p = np.random.permutation(Nval)
    Xval = Xval[p, :]
    yval = yval[p]

    p = np.random.permutation(Ntest)
    Xtest = Xtest[p, :]
    ytest = ytest[p]

    return Xtrain, Xval, Xtest, ytrain, yval, ytest


def plot_prior(model, n, epoch, n_latent, d=28, prefix='', suffix=''):
    # https://keras.io/examples/generative/vae/

    if n_latent == 2:
        norm = tfd.Normal(0, 1)
        grid_x = norm.quantile(np.linspace(0.01, 0.99, n))
        # reverse the y-grid due to imshow
        grid_y = norm.quantile(np.linspace(0.99, 0.01, n))
        canvas1 = np.zeros((n * d, n * d))
        canvas2 = np.zeros((n * d, n * d))
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z = np.array([[xi, yi]])
                logit, pxz = model.decoder(z)
                x_samples = pxz.sample().numpy().squeeze()
                x_decoded = tf.nn.sigmoid(logit)
                digit = tf.reshape(x_decoded[0], (d, d))
                canvas1[j * d: (j + 1) * d, i * d: (i + 1) * d] = digit.numpy()
                canvas2[j * d: (j + 1) * d, i * d: (i + 1) * d] = x_samples[i * n + j, :].reshape(28, 28)
    else:
        norm = tfd.Normal(np.zeros(n_latent), np.ones(n_latent))
        z = norm.sample(n * n)
        logits, pxz = model.decoder(z[None, :, :])
        x_samples = pxz.sample().numpy().squeeze()
        x_decoded = tf.nn.sigmoid(logits.numpy().squeeze())
        canvas1 = np.zeros((n * d, n * d))
        canvas2 = np.zeros((n * d, n * d))

        for i in range(n):
            for j in range(n):
                digit = tf.reshape(x_decoded[i * n + j], (d, d))
                canvas1[j * d: (j + 1) * d, i * d: (i + 1) * d] = digit.numpy()
                canvas2[j * d: (j + 1) * d, i * d: (i + 1) * d] = x_samples[i * n + j, :].reshape(28, 28)

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas1, cmap='gray_r')
    plt.axis('Off')
    plt.savefig('task01_latent_epoch_{0:02d}'.format(epoch) + suffix)
    plt.close()

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas2, cmap='gray_r')
    plt.axis('Off')
    plt.savefig(prefix + 'latent_samples_epoch_{0:02d}'.format(epoch) + suffix)
    plt.close()


class MyMetric():
    def __init__(self):
        self.VALUES = []
        self.N = []

    def update_state(self, losses):
        self.VALUES.append(losses)
        self.N.append(losses.shape[0])

    def result(self):
        VALUES = tf.concat(self.VALUES, axis=0)
        return tf.reduce_sum(VALUES) / tf.cast(tf.reduce_sum(self.N), tf.float32)

    def reset_states(self):
        self.VALUES = []
        self.N = []


class MyMetric2():
    def __init__(self):
        self.VALUES = []

    def update_state(self, losses):
        self.VALUES.append(losses)

    def result(self):
        return tf.reduce_mean(self.VALUES, axis=0)

    def reset_states(self):
        self.VALUES = []


def generate_and_save_images(model, z2, epoch, string):

    # ---- plot settings
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['font.size'] = 15.0
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['figure.autolayout'] = True

    x_samples, x_probs = model.sample(z2)
    x_samples = x_samples.numpy().squeeze()
    x_probs = x_probs.numpy().squeeze()

    n = int(np.sqrt(x_samples.shape[0]))

    canvas = np.zeros((n * 28, 2 * n * 28))

    for i in range(n):
        for j in range(n):
            canvas[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_samples[i * n + j].reshape(28, 28)
            canvas[i * 28: (i + 1) * 28, n * 28 + j * 28: n * 28 + (j + 1) * 28] = x_probs[i * n + j].reshape(28, 28)

    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.imshow(canvas, cmap='gray_r')
    plt.title("epoch {:04d}".format(epoch))
    plt.axis('off')
    plt.savefig(string + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

