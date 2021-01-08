import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
sys.path.insert(0, '../src')
import iwae1
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# TODO: match task01 experiment settings
n_hidden = 200
n_latent = 2
string = "task01_{0}_{1}_{2}".format("iwae_elbo", 1, 50)
n_examples = 20
L = 1000

# ---- load model
model = iwae1.IWAE(n_hidden, n_latent)
model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- load data
(Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

Xtest = Xtest.reshape(Ntest, -1) / 255
Xtest = utils.bernoullisample(Xtest)


def get_grid(range1, range2, n1, n2):
    x = np.linspace(range1[0], range1[1], n1)
    y = np.linspace(range2[0], range2[1], n2)

    X, Y = np.meshgrid(x, y)
    grid = np.asarray([X.reshape(n1 * n2), Y.reshape(n1 * n2)]).T

    return grid, X, Y


# ---- prepare contours over a grid
n1 = 200
n2 = 200
range_ = [-3, 3]

grid, X, Y = get_grid(range_, range_, n1, n2)

# ---- prior over the grid
prior = multivariate_normal(np.zeros(n_latent), np.ones(n_latent))
prior_pdf = prior.pdf(grid).reshape(n2, n1)

for i, x in enumerate(Xtest[:n_examples, :]):

    # ---- plot digit
    plt.clf()
    fig, ax = plt.subplots()
    ax.imshow(Xtest[i, :].reshape(28, 28), cmap='gray_r')
    ax.axis('off')
    plt.savefig('../results/{0}_digit_{1}'.format(string, i))
    plt.close()

    # ---- find the variational posterior
    res = model(x[None, :], L)
    z = res["z"]
    snis_z = res["snis_z"]
    # al = res["al"]
    _, qzx = model.encoder(x[None, :], 1)
    q_mu = qzx.loc[0].numpy()
    q_std = qzx.scale[0].numpy()

    # ---- then make a grid near the variational posterior
    scale = 2
    range_x = [np.max([-3, q_mu[0] - scale * q_std[0]]),
              np.min([3, q_mu[0] + scale * q_std[0]])]
    range_y = [np.max([-3, q_mu[1] - scale * q_std[1]]),
              np.min([3, q_mu[1] + scale * q_std[1]])]

    grid1, X1, Y1 = get_grid(range_x, range_y, n1, n2)

    # ---- feed the grid through the decoder
    _, pxz = model.decoder(grid1[None, :, :])

    # ---- evaluate the grid in the prior
    lpz = prior.logpdf(grid1)

    # ---- evaluate the digit in the observation model, over the grid
    lpxz = tf.reduce_sum(pxz.log_prob(x[None, :]), axis=-1).numpy().squeeze()

    unnormalized_log_posterior = lpxz + lpz

    log_posterior = unnormalized_log_posterior - logsumexp(unnormalized_log_posterior)

    log_posterior = log_posterior.reshape(n2, n1)

    # ---- true posterior
    plt.clf()
    fig, ax = plt.subplots()
    ax.contour(X, Y, prior_pdf, 5, cmap='RdGy_r', linewidths=1)
    # ax.contour(X1, Y1, np.exp(log_posterior) + 1e-16, 5, cmap='RdGy_r', linewidths=1)
    ax.imshow(np.exp(log_posterior), cmap='gray_r',
              extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
              origin='lower')
    ax.axis('equal')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.set_xlim([range_x[0], range_x[1]])
    ax.set_ylim([range_y[0], range_y[1]])
    plt.savefig('../results/{0}_true_posterior_{1}'.format(string, i))
    plt.close()

