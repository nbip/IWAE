import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow_probability import distributions as tfd
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt


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



def plot_prior(model, n, epoch, n_latent, d=28, suffix=''):
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
    plt.savefig('task01_latent_samples_epoch_{0:02d}'.format(epoch) + suffix)
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
