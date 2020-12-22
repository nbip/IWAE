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

# ---- plot settings
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 15.0
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['figure.autolayout'] = True

# ---- set random seeds
np.random.seed(123)
tf.random.set_seed(123)

# ---- number of passes over the data, see bottom of page 6 in [1]
if args.epochs == -1:
    epochs = 0
    learning_rate_change_epoch = []
    learning_rates = []

    for i in range(8):
        learning_rates.append(0.001 * 10**(-i/7))
        epochs += 3 ** i
        learning_rate_change_epoch.append(epochs)

else:
    epochs = args.epochs
    learning_rates = []
    learning_rates.append(0.0001)


# ---- experiment settings
n_latent1 = args.n_latent1
n_latent2 = args.n_latent2
n_samples = args.n_samples
n_hidden1 = args.n_hidden1
n_hidden2 = args.n_hidden2
batch_size = args.batch_size

(Xtrain, ytrain), (Xval, yval), (Xtest, ytest) = utils.dynamic_binarization_mnist()
Ntrain, D = Xtrain.shape
Nval, _ = Xval.shape
Ntest, _ = Xtest.shape

x = Xtrain[:batch_size, :]

# ---- train step
@tf.function
def train_step(model, x, n_samples, beta, optimizer):

    with tf.GradientTape() as tape:
        res = model(x, n_samples, beta)
        loss = res["loss"]

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return res


# ---- val step
@tf.function
def val_step(model, x, beta, n_samples):
    return model(x, n_samples, beta)


# ---- prepare tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "/tmp/iwae/task13/" + current_time + "/train"
val_log_dir = "/tmp/iwae/task13/" + current_time + "/val"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

steps_pr_epoch = Ntrain // batch_size
total_steps = steps_pr_epoch * epochs

# ---- instantiate the model, optimizer and metrics
model = iwae.IWAE(n_hidden1,
                  n_hidden2,
                  n_latent1,
                  n_latent2)

optimizer = keras.optimizers.Adam(learning_rates[0], epsilon=1e-4)
print("Initial learning rate: ", optimizer.learning_rate.numpy())

# ---- binarize the validation data
# ---- we'll only do this once, while the training data is binarized at the
# ---- start of each epoch
Xval = utils.bernoullisample(Xval)

val_dataset = (tf.data.Dataset.from_tensor_slices(Xval)
               .shuffle(Nval).batch(1000))

val_elbo_metric = utils.MyMetric2()
val_elbo2_metric = utils.MyMetric2()
val_lpxz1_metric = utils.MyMetric()
val_lpz1z2_metric = utils.MyMetric()
val_lpz2_metric = utils.MyMetric()
val_lqz1x_metric = utils.MyMetric()
val_lqz2z1_metric = utils.MyMetric()

# ---- do the training
start = time.time()
best = float(-np.inf)

train_history = []

for epoch in range(epochs):

    # ---- binarize the training data at the start of each epoch
    Xtrain_binarized = utils.bernoullisample(Xtrain)

    train_dataset = (tf.data.Dataset.from_tensor_slices(Xtrain_binarized)
        .shuffle(Ntrain).batch(batch_size))

    # ---- check if the learning rate needs to be updated
    if args.epochs == -1 and np.sum(epoch == np.asarray(learning_rate_change_epoch)) > 0:
        idx = np.where(epoch == np.asarray(learning_rate_change_epoch))[0][0]

        new_learning_rate = learning_rates[idx + 1]
        old_learning_rate = optimizer.learning_rate.numpy()

        print("Changing learning rate from {0} to {1}".format(old_learning_rate, new_learning_rate))
        optimizer.learning_rate.assign(new_learning_rate)

    for _step, x_batch in enumerate(train_dataset):
        step = _step + steps_pr_epoch * epoch

        beta = np.min([step / 200000, 1.0]).astype(np.float32)

        # ---- one training step
        res = train_step(model, x_batch, n_samples, beta, optimizer)

        if step % 200 == 0:
            train_history.append(res["elbo"].numpy().mean())

            # ---- write training stats to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('Evaluation/elbo', res["elbo"], step=step)
                tf.summary.scalar('Evaluation/elbo2', res["elbo2"], step=step)
                tf.summary.scalar('Evaluation/lpxz1', res['lpxz1'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lpz1z2', res['lpz1z2'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lqz1x', res['lqz1x'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lqz2z1', res['lqz2z1'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lpz2', res['lpz2'].numpy().mean(), step=step)

            # ---- collect validation stats
            for x_val_batch in val_dataset:
                val_res = val_step(model, x_val_batch, n_samples, beta)

                val_elbo_metric.update_state(val_res["elbo"])
                val_elbo2_metric.update_state(val_res["elbo2"])
                val_lpxz1_metric.update_state(val_res['lpxz1'])
                val_lpz1z2_metric.update_state(val_res['lpz1z2'])
                val_lpz2_metric.update_state(val_res['lpz2'])
                val_lqz1x_metric.update_state(val_res['lqz1x'])
                val_lqz2z1_metric.update_state(val_res['lqz2z1'])

            # ---- summarize the results over the batches and reset the metrics
            val_elbo = val_elbo_metric.result()
            val_elbo_metric.reset_states()
            val_elbo2 = val_elbo2_metric.result()
            val_elbo2_metric.reset_states()
            val_lpxz1 = val_lpxz1_metric.result()
            val_lpxz1_metric.reset_states()
            val_lpz1z2 = val_lpz1z2_metric.result()
            val_lpz1z2_metric.reset_states()
            val_lpz2 = val_lpz2_metric.result()
            val_lpz2_metric.reset_states()
            val_lqz1x = val_lqz1x_metric.result()
            val_lqz1x_metric.reset_states()
            val_lqz2z1 = val_lqz2z1_metric.result()
            val_lqz2z1_metric.reset_states()

            # ---- write val stats to tensorboard
            with val_summary_writer.as_default():
                tf.summary.scalar('Evaluation/elbo', val_elbo, step=step)
                tf.summary.scalar('Evaluation/elbo2', val_elbo2, step=step)
                tf.summary.scalar('Evaluation/lpxz1', val_lpxz1, step=step)
                tf.summary.scalar('Evaluation/lpz1z2', val_lpz1z2, step=step)
                tf.summary.scalar('Evaluation/lqz1x', val_lqz1x, step=step)
                tf.summary.scalar('Evaluation/lqz2z1', val_lqz2z1, step=step)
                tf.summary.scalar('Evaluation/lpz2', val_lpz2, step=step)
                tf.summary.scalar('Evaluation/beta', beta, step=step)

            # ---- save the model if the validation loss improves
            if val_elbo > best:
                print("saving model...")
                model.save_weights('/tmp/iwae/task13/best_weights' + '_nsamples_{}'.format(n_samples))
                best = val_elbo

            took = time.time() - start
            start = time.time()

            print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                  .format(epoch, epochs, step, total_steps, res["elbo"].numpy(), val_elbo.numpy(), took))

# ---- save final weights
model.save_weights('/tmp/iwae/task13/final_weights' + '_nsamples_{}'.format(n_samples))

# ---- test-set llh estimate using 5000 samples
test_elbo_metric = utils.MyMetric()
L = 5000

# ---- binarize Xtest
Xtest = utils.bernoullisample(Xtest)

# ---- since we are using 5000 importance samples we have to loop over each element of the test-set
for i, x in enumerate(Xtest):
    res = model(x[None, :].astype(np.float32), L)
    test_elbo_metric.update_state(res['elbo'][None, None])
    if i % 200 == 0:
        print("{0}/{1}".format(i, Ntest))

test_set_llh = test_elbo_metric.result()
test_elbo_metric.reset_states()

print("Test-set {0} sample log likelihood estimate: {1:.4f}".format(L, test_set_llh))

# ---- plot samples from the prior


