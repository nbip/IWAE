import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow import keras
import numpy as np
import os
import argparse
import datetime
import time
import sys
sys.path.insert(0, './src')
import utils
import iwae1
import iwae2

parser = argparse.ArgumentParser()
parser.add_argument("--stochastic_layers", type=int, default=1, choices=[1, 2], help="number of stochastic layers in the model")
parser.add_argument("--n_samples", type=int, default=5, help="number of importance samples")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--epochs", type=int, default=-1,
                    help="numper of epochs, if set to -1 number of epochs "
                         "will be set based on the learning rate scheme from the paper")
parser.add_argument("--objective", type=str, default="iwae_elbo", choices=["vae_elbo", "iwae_elbo", "iwae_eq14", "vae_elbo_kl"])
parser.add_argument("--gpu", type=str, default='0', help="Choose GPU")
args = parser.parse_args()
print(args)

# ---- string describing the experiment, to use in tensorboard and plots
string = "task01_{0}_{1}_{2}".format(args.objective, args.stochastic_layers, args.n_samples)

# ---- set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ---- dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ---- set random seeds
np.random.seed(123)
tf.random.set_seed(123)

# ---- number of passes over the data, see bottom of page 6 in [1]
if args.epochs == -1:
    epochs = 0
    learning_rate_dict = {}

    for i in range(8):
        learning_rate = 0.001 * 10**(-i/7)
        learning_rate_dict[epochs] = learning_rate
        epochs += 3 ** i

else:
    epochs = args.epochs
    learning_rate_dict = {}
    learning_rate_dict[0] = 0.0001

# ---- load data
(Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

# ---- reshape to vectors
Xtrain = Xtrain.reshape(Ntrain, -1) / 255
Xtest = Xtest.reshape(Ntest, -1) / 255

# ---- experiment settings
objective = args.objective
n_samples = args.n_samples
batch_size = args.batch_size
steps_pr_epoch = Ntrain // batch_size
total_steps = steps_pr_epoch * epochs

# ---- prepare tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "/tmp/iwae/{0}/".format(string) + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_log_dir = "/tmp/iwae/{0}/".format(string) + current_time + "/test"
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# ---- instantiate the model, optimizer and metrics
if args.stochastic_layers == 1:
    n_latent = [2]
    n_hidden = [200]
    model = iwae1.IWAE(n_hidden[0], n_latent[0])
else:
    n_latent = [2, 2]
    n_hidden = [200, 100]
    model = iwae2.IWAE(n_hidden, n_latent)

optimizer = keras.optimizers.Adam(learning_rate_dict[0], epsilon=1e-4)
print("Initial learning rate: ", optimizer.learning_rate.numpy())

# ---- prepare plotting of samples during training
# use the same samples from the prior throughout training
pz = tfd.Normal(0, 1)
z = pz.sample([100, n_latent[-1]])

plt_epochs = list(2**np.arange(12))
plt_epochs.insert(0, 0)
plt_epochs.append(epochs-1)

# ---- binarize the test data
# we'll only do this once, while the training data is binarized at the
# start of each epoch
Xtest = utils.bernoullisample(Xtest)

# ---- do the training
start = time.time()
best = float(-np.inf)

for epoch in range(epochs):

    # ---- binarize the training data at the start of each epoch
    Xtrain_binarized = utils.bernoullisample(Xtrain)

    train_dataset = (tf.data.Dataset.from_tensor_slices(Xtrain_binarized)
        .shuffle(Ntrain).batch(batch_size))

    # ---- plot samples from the prior at this epoch
    if epoch in plt_epochs:
        model.generate_and_save_images(z, epoch, string)
        model.generate_and_save_posteriors(Xtest, ytest, 10, epoch, string)

    # ---- check if the learning rate needs to be updated
    if args.epochs == -1 and epoch in learning_rate_dict:
        new_learning_rate = learning_rate_dict[epoch]
        old_learning_rate = optimizer.learning_rate.numpy()

        print("Changing learning rate from {0} to {1}".format(old_learning_rate, new_learning_rate))
        optimizer.learning_rate.assign(new_learning_rate)

    for _step, x_batch in enumerate(train_dataset):
        step = _step + steps_pr_epoch * epoch

        # ---- warm-up
        beta = 1.0
        # beta = np.min([step / 200000, 1.0]).astype(np.float32)

        # ---- one training step
        res = model.train_step(x_batch, n_samples, beta, optimizer, objective=objective)

        if step % 200 == 0:

            # ---- write training stats to tensorboard
            with train_summary_writer.as_default():
                model.write_to_tensorboard(res, step)

            # ---- monitor the test-set
            test_res = model.val_step(Xtest, n_samples, beta)

            # ---- write test stats to tensorboard
            with test_summary_writer.as_default():
                model.write_to_tensorboard(test_res, step)

            took = time.time() - start
            start = time.time()

            print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                  .format(epoch, epochs, step, total_steps, res[objective].numpy(), test_res[objective], took))

# ---- save final weights
model.save_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- load the final weights?
# model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- test-set llh estimate using 5000 samples
test_elbo_metric = utils.MyMetric()
L = 5000

# ---- since we are using 5000 importance samples we have to loop over each element of the test-set
for i, x in enumerate(Xtest):
    res = model(x[None, :].astype(np.float32), L)
    test_elbo_metric.update_state(res['iwae_elbo'][None, None])
    if i % 200 == 0:
        print("{0}/{1}".format(i, Ntest))

test_set_llh = test_elbo_metric.result()
test_elbo_metric.reset_states()

print("Test-set {0} sample log likelihood estimate: {1:.4f}".format(L, test_set_llh))

# TODO: compare the variational posterior to the true posterior over a grid
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))


def get_grid(range1, range2, n1, n2):
    x = np.linspace(range1[0], range1[1], n1)
    y = np.linspace(range2[0], range2[1], n2)

    X, Y = np.meshgrid(x, y)
    grid = np.asarray([X.reshape(n1 * n2), Y.reshape(n1 * n2)]).T

    return grid, X, Y


# ---- fontsize and markersize
fs = 35
ms = 100
n_examples = 10

# ---- get some results on the test-set
res = model(Xtest, 20)

z, QZX = model.encoder(Xtest, 20)

# ---- prepare contours over a grid
n1 = 200
n2 = 200
range_ = [-3, 3]

grid, X, Y = get_grid(range_, range_, n1, n2)

# ---- prior over the grid
prior = multivariate_normal(np.zeros(n_latent), np.ones(n_latent))
prior_pdf = prior.pdf(grid).reshape(n2, n1)

for i in range(n_examples):

    # ---- test-data
    plt.clf()
    fig, ax = plt.subplots()
    ax.imshow(Xtest[i, :].reshape(28, 28), cmap='gray_r')
    ax.axis('off')
    plt.savefig('{0}_digit_{1}'.format(string, i))
    plt.close()

    # ---- new grid scaled to the variational posterior
    qzx = QZX[i, :]

    scale = 3
    range1 = [(qzx.loc[0] - scale * qzx.scale[0]).numpy(), (qzx.loc[0] + scale * qzx.scale[0]).numpy()]
    range2 = [(qzx.loc[1] - scale * qzx.scale[1]).numpy(), (qzx.loc[1] + scale * qzx.scale[1]).numpy()]

    grid1, X1, Y1 = get_grid(range1, range2, n1, n2)

    variational_posterior = np.exp(tf.reduce_sum(qzx.log_prob(grid1), axis=-1).numpy().reshape(n2, n1))

    plt.clf()
    fig, ax = plt.subplots()
    ppr = ax.contour(X, Y, prior_pdf, 5)
    ppo = ax.contour(X1, Y1, variational_posterior, 5, cmap='RdGy_r', linewidths=1)
    ax.axis('equal')
    plt.savefig('{0}_variational_posterior_{1}'.format(string, i))
    plt.close()

    # ---- reconstructions
    logits, pxz = model.decoder(z[:, i, :][:, None, :])
    x_rec = pxz.sample().numpy().squeeze()

    # put the original digit first
    canvas = np.zeros((28, 28 * (1 + x_rec.shape[0])))
    canvas[:, :28] = Xtest[i, :].reshape(28, 28)

    for j in range(1, x_rec.shape[0] + 1):
        canvas[:, j * 28: (j + 1) * 28] = x_rec[j - 1, :].reshape(28, 28)

    plt.clf()
    fig, ax = plt.subplots()
    ax.imshow(canvas, cmap='gray_r')
    ax.axis('off')
    plt.savefig('{0}_reconstructions_{1}'.format(string, i))
    plt.close()

    # ---- inspect the true posterior
    # prior p(z)
    lpz = prior.logpdf(grid1)

    # observation model p(x|z) over the grid
    _, observation_model = model.decoder(grid1[None, :, :])

    lpxz = tf.reduce_sum(observation_model.log_prob(Xtest[i, :]), axis=-1).numpy().squeeze()

    # log posterior and normalization
    log_posterior = lpxz + lpz

    log_posterior = log_posterior - logsumexp(log_posterior)

    log_posterior = log_posterior.reshape(n2, n1)

    plt.clf()
    fig, ax = plt.subplots()
    ax.contour(X, Y, prior_pdf, 5)
    ax.contour(X1, Y1, np.exp(log_posterior), 5, cmap='RdGy_r', linewidths=1)
    ax.axis('equal')
    plt.savefig('{0}_true_posterior_{1}'.format(string, i))
    plt.close()

    plt.clf()
    fig, ax = plt.subplots()
    ax.contour(X1, Y1, variational_posterior, 5, colors='black', linewidths=1)
    ax.contour(X1, Y1, np.exp(log_posterior), 5, cmap='RdGy_r', linewidths=1)
    ax.axis('equal')
    # get axis limits for last plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.savefig('{0}_posteriors_{1}'.format(string, i))
    plt.close()

    plt.clf()
    fig, ax = plt.subplots()
    ax.contour(X, Y, prior_pdf, 5)
    ax.contour(X1, Y1, variational_posterior, 5, colors='black', linewidths=1)
    ax.contour(X1, Y1, np.exp(log_posterior), 5, cmap='RdGy_r', linewidths=1)
    ax.axis('equal')
    plt.savefig('{0}_posteriorss_{1}'.format(string, i))
    plt.close()

    plt.clf()
    fig, ax = plt.subplots()
    ax.contour(X, Y, prior_pdf, 5)
    ax.contour(X1, Y1, np.exp(log_posterior), 5, cmap='RdGy_r', linewidths=1)
    ax.axis('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig('{0}_posteriorsss_{1}'.format(string, i))
    plt.close()
