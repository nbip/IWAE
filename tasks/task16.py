import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow import keras
import numpy as np
import os
import argparse
import datetime
import time
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '/home/nbip/proj/python/python-TF2/tf2-01/notebooks/IWAE/src')
import utils
import iwae

parser = argparse.ArgumentParser()
parser.add_argument("--n_latent1", type=int, default=100, help="number of latent space dimensions in 1st stochastic layer")
parser.add_argument("--n_latent2", type=int, default=50, help="number of latent space dimensions in 2nd stochastic layer")
parser.add_argument("--n_samples", type=int, default=5, help="number of importance samples")
parser.add_argument("--n_hidden1", type=int, default=200, help="number of hidden units in the first layer")
parser.add_argument("--n_hidden2", type=int, default=100, help="number of hidden units in the first layer")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--epochs", type=int, default=-1,
                    help="numper of epochs, if set to -1 number of epochs "
                         "will be set based on the learning rate scheme from the paper")
parser.add_argument("--gpu", type=str, default='0', help="Choose GPU")
args = parser.parse_args()
print(args)

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
n_latent1 = args.n_latent1
n_latent2 = args.n_latent2
n_samples = args.n_samples
n_hidden1 = args.n_hidden1
n_hidden2 = args.n_hidden2
batch_size = args.batch_size
steps_pr_epoch = Ntrain // batch_size
total_steps = steps_pr_epoch * epochs

# ---- prepare tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "/tmp/iwae/task16/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# ---- prepare plotting of samples during training
# use the same samples from the prior throughout training
pz2 = tfd.Normal(0, 1)
z2 = pz2.sample([100, n_latent2])

plt_epochs = list(2**np.arange(12))
plt_epochs.insert(0, 0)
plt_epochs.append(epochs)

# ---- instantiate the model and optimizer
model = iwae.IWAE(n_hidden1,
                  n_hidden2,
                  n_latent1,
                  n_latent2)

optimizer = keras.optimizers.Adam(learning_rate_dict[0], epsilon=1e-4)
print("Initial learning rate: ", optimizer.learning_rate.numpy())

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
        utils.generate_and_save_images(model, z2, epoch, "task16_{}_".format(n_samples))

    # ---- check if the learning rate needs to be updated
    if args.epochs == -1 and epoch in learning_rate_dict:

        new_learning_rate = learning_rate_dict[epoch]
        old_learning_rate = optimizer.learning_rate.numpy()

        print("Changing learning rate from {0} to {1}".format(old_learning_rate, new_learning_rate))
        optimizer.learning_rate.assign(new_learning_rate)

    for _step, x_batch in enumerate(train_dataset):
        step = _step + steps_pr_epoch * epoch

        # warm-up
        beta = np.min([step / 200000, 1.0]).astype(np.float32)

        # ---- one training step
        res = model.train_step(x_batch, n_samples, beta, optimizer)

        if step % 200 == 0:

            # ---- write training stats to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('Evaluation/vae_elbo', res["vae_elbo"], step=step)
                tf.summary.scalar('Evaluation/beta_vae_elbo', res["beta_vae_elbo"], step=step)
                tf.summary.scalar('Evaluation/iwae_elbo', res["iwae_elbo"], step=step)
                tf.summary.scalar('Evaluation/beta_iwae_elbo', res["beta_iwae_elbo"], step=step)
                tf.summary.scalar('Evaluation/lpxz1', res['lpxz1'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lpz1z2', res['lpz1z2'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lqz1x', res['lqz1x'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lqz2z1', res['lqz2z1'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lpz2', res['lpz2'].numpy().mean(), step=step)

            took = time.time() - start
            start = time.time()

            print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, time: {5:.2f}"
                  .format(epoch, epochs, step, total_steps, res["iwae_elbo"].numpy(), took))

# ---- save final weights
model.save_weights('/tmp/iwae/task16/final_weights' + '_nsamples_{}'.format(n_samples))

# ---- load the final weights?
# model.load_weights('/tmp/iwae/task16/final_weights' + '_nsamples_{}'.format(n_samples))

# ---- test-set llh estimate using 5000 samples
test_elbo_metric = utils.MyMetric()
L = 5000

# ---- binarize Xtest
Xtest = utils.bernoullisample(Xtest)

# ---- since we are using 5000 importance samples we have to loop over each element of the test-set
for i, x in enumerate(Xtest):
    res = model(x[None, :].astype(np.float32), L)
    test_elbo_metric.update_state(res['iwae_elbo'][None, None])
    if i % 200 == 0:
        print("{0}/{1}".format(i, Ntest))

test_set_llh = test_elbo_metric.result()
test_elbo_metric.reset_states()

print("Test-set {0} sample log likelihood estimate: {1:.4f}".format(L, test_set_llh))

