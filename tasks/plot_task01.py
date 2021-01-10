import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_probability import distributions as tfd
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy.special import softmax
sys.path.insert(0, '../src')
print(os.getcwd())
import iwae1
import utils


def plot(model, Xtest, string):

    def get_grid(range1, range2, n1, n2):
        x = np.linspace(range1[0], range1[1], n1)
        y = np.linspace(range2[0], range2[1], n2)

        X, Y = np.meshgrid(x, y)
        grid = np.asarray([X.reshape(n1 * n2), Y.reshape(n1 * n2)]).T

        return grid, X, Y

    # ---- prior over the grid
    prior = multivariate_normal(np.zeros(n_latent), np.ones(n_latent))

    for i, x in enumerate(Xtest[:n_examples, :]):

        # ---- find the variational posterior
        res = model(x[None, :], L)

        z = res["z"].numpy().squeeze()

        _, qzx = model.encoder(x[None, :], 1)
        q_mu = qzx.loc[0].numpy()
        q_std = qzx.scale[0].numpy()
        qzx = tfd.Normal(q_mu, q_std)

        # ---- then make a grid near the variational posterior
        n1 = 200
        n2 = 200
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

        # ---- true posterior
        unnormalized_log_posterior = lpxz + lpz

        log_posterior = unnormalized_log_posterior - logsumexp(unnormalized_log_posterior)

        log_posterior = log_posterior.reshape(n2, n1)

        # ---- variational posterior
        variational_posterior = tf.reduce_sum(qzx.log_prob(grid1), axis=-1).numpy().reshape(n2, n1)

        # ---- sampling importance resampling
        n_samples = 200
        # al = res["al"].numpy()
        lpxz = res["lpxz"].numpy().squeeze()
        lqzx = res["lqzx"].numpy().squeeze()
        lpz = res["lpz"].numpy().squeeze()

        log_w = lpxz + lpz - lqzx

        # self-normalized importance weights
        al = softmax(log_w, axis=0)

        # sample from the z-samples according to the importance weights
        idx = np.random.choice(np.arange(0, z.shape[0]), size=n_samples, replace=False, p=al)
        z_sir = z[idx]

        # ---- reconstructions using regular samples from the variational posterior
        logits = res["logits"].numpy().squeeze()
        pxz = tfd.Bernoulli(logits=logits)
        x_samples = pxz.sample().numpy()

        n = 5
        canvas1 = np.zeros((n * 28, n * 28))

        for j in range(n):
            for k in range(n):
                canvas1[j * 28: (j + 1) * 28, k * 28: (k + 1) * 28] = x_samples[j * n + k].reshape(28, 28)

        # ---- reconstructions using SIR samples
        x_samples_sir = x_samples[idx]

        canvas2 = np.zeros((n * 28, n * 28))

        for j in range(n):
            for k in range(n):
                canvas2[j * 28: (j + 1) * 28, k * 28: (k + 1) * 28] = x_samples_sir[j * n + k].reshape(28, 28)

        # ---- plot it all
        plt.clf()
        fig, ax = plt.subplots(2,4, figsize=(20, 8))

        # ---- original digit
        ax[0, 0].imshow(Xtest[i, :].reshape(28, 28), cmap='gray_r')
        ax[0, 0].set_title('original digit', fontsize=fs)
        ax[0, 0].axis('off')

        # ---- true and variational posterior
        ax[0, 1].contour(X1, Y1, np.exp(variational_posterior) + 1e-16, 5, cmap='Purples', linewidths=3)
        ax[0, 1].contour(X1, Y1, np.exp(log_posterior) + 1e-16, 5, cmap='RdGy_r', linewidths=3)
        ax[0, 1].imshow(np.exp(log_posterior), cmap='gray_r',
                  extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                  origin='lower')
        ax[0, 1].axis('equal')
        ax[0, 1].spines['top'].set_visible(True)
        ax[0, 1].spines['right'].set_visible(True)
        ax[0, 1].set_xlim([range_x[0], range_x[1]])
        ax[0, 1].set_ylim([range_y[0], range_y[1]])
        ax[0, 1].set_title('true and variational \nposterior', fontsize=fs)

        # ---- samples from the variational posterior
        ax[0, 2].contour(X1, Y1, np.exp(variational_posterior) + 1e-16, 5, cmap='Purples', linewidths=3)
        ax[0, 2].imshow(np.exp(log_posterior), cmap='gray_r',
                  extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                  origin='lower')
        ax[0, 2].scatter(z[:, 0], z[:, 1], marker='.', alpha=0.3, color="Purple")
        ax[0, 2].axis('equal')
        ax[0, 2].spines['top'].set_visible(True)
        ax[0, 2].spines['right'].set_visible(True)
        ax[0, 2].set_xlim([range_x[0], range_x[1]])
        ax[0, 2].set_ylim([range_y[0], range_y[1]])
        ax[0, 2].set_title('variational posterior \nsamples', fontsize=fs)

        # ---- sampling importance resampling
        ax[1, 2].contour(X1, Y1, np.exp(variational_posterior) + 1e-16, 5, cmap='Purples', linewidths=3)
        ax[1, 2].imshow(np.exp(log_posterior), cmap='gray_r',
                  extent=[range_x[0], range_x[1], range_y[0], range_y[1]],
                  origin='lower')
        ax[1, 2].scatter(z_sir[:n_samples, 0], z_sir[:n_samples, 1], marker='.', alpha=0.3, color="Purple")
        ax[1, 2].axis('equal')
        ax[1, 2].spines['top'].set_visible(True)
        ax[1, 2].spines['right'].set_visible(True)
        ax[1, 2].set_xlim([range_x[0], range_x[1]])
        ax[1, 2].set_ylim([range_y[0], range_y[1]])
        ax[1, 2].set_title('SIR', fontsize=fs)

        # ---- variational posterior sample reconstructions
        ax[0, 3].imshow(canvas1, cmap='gray_r')
        ax[0, 3].axis('off')
        ax[0, 3].set_title('reconstrictions', fontsize=fs)

        # ---- SIR reconstructions
        ax[1, 3].imshow(canvas2, cmap='gray_r')
        ax[1, 3].axis('off')
        # ax[1, 3].set_title('Reconstructions of SIR samples', fontsize=fs)

        ax[1, 0].axis('off')
        ax[1, 1].axis('off')

        plt.savefig('../results/{0}_all_in_one_{1}.png'.format(string, i))
        plt.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    n_hidden = 200
    n_latent = 2
    string = "task01_{0}_{1}_{2}".format("iwae_elbo", 1, 50)
    n_examples = 20
    L = 10000
    fs = 25  # fontsize
    lw = 3   # line width

    # ---- set random seeds
    np.random.seed(123)
    tf.random.set_seed(123)

    # ---- load model
    model = iwae1.IWAE(n_hidden, n_latent)
    model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))

    # ---- load data
    _, (Xtest, ytest) = keras.datasets.mnist.load_data()
    Ntest = Xtest.shape[0]

    Xtest = Xtest.reshape(Ntest, -1) / 255
    Xtest = utils.bernoullisample(Xtest)

    plot(model, Xtest, string)
