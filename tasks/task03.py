import tensorflow as tf
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
import utils
import model

parser = argparse.ArgumentParser()
parser.add_argument("--n_latent", type=int, default=50, help="number of latent space dimensions")
parser.add_argument("--n_samples", type=int, default=5, help="number of importance samples")
parser.add_argument("--n_hidden", type=int, default=200, help="number of hidden units in the NN layers")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--epochs", type=int, default=-1,
                    help="numper of epochs, if set to -1 number of epochs "
                         "will be set based on the learning rate scheme from the paper")
parser.add_argument("--gpu", type=str, default='0', help="Choose GPU")
args = parser.parse_args()

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
n_latent = args.n_latent
n_samples = args.n_samples
n_hidden = args.n_hidden
batch_size = args.batch_size

# ---- load data
Xtrain, Xval, Xtest, ytrain, yval, ytest = utils.static_binarization_mnist()


# ---- train and validation steps
@tf.function
def train_step(model, x, n_samples, optimizer):

    with tf.GradientTape() as tape:
        res = model(x, n_samples)
        loss = res["loss"]

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return res


@tf.function
def val_step(model, x, n_samples):
    return model(x, n_samples)


# ---- prepare tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "/tmp/iwae/task03/" + current_time + "/train"
val_log_dir = "/tmp/iwae/task03/" + current_time + "/val"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

# ---- prepare the data
Ntrain = Xtrain.shape[0]
Nval = Xval.shape[0]
Ntest = Xtest.shape[0]
steps_pr_epoch = Ntrain // batch_size
total_steps = steps_pr_epoch * epochs

train_dataset = (tf.data.Dataset.from_tensor_slices(Xtrain)
                 .shuffle(Ntrain).batch(batch_size))
val_dataset = (tf.data.Dataset.from_tensor_slices(Xval)
               .shuffle(Nval).batch(1000))


# ---- instantiate the model, optimizer and metrics
model = model.VAE(n_hidden, n_latent)

optimizer = keras.optimizers.Adam(learning_rates[0])
print("Initial learning rate: ", optimizer.learning_rate.numpy())

# ---- Use the MyMetric class to collect losses over several batches
# ---- If the whole validation set can fit in GPU memory then this is not needed
val_elbo_metric = utils.MyMetric()
val_lpxz_metric = utils.MyMetric()
val_lpz_metric = utils.MyMetric()
val_lqzx_metric = utils.MyMetric()
val_kl_metric = utils.MyMetric()

# ---- do the training
start = time.time()
best = float(-np.inf)

train_history = []
val_history = []

for epoch in range(epochs):

    # ---- check if the learning rate needs to be updated
    if args.epochs == -1 and np.sum(epoch == np.asarray(learning_rate_change_epoch)) > 0:
        idx = np.where(epoch == np.asarray(learning_rate_change_epoch))[0][0]

        new_learning_rate = learning_rates[idx + 1]
        old_learning_rate = optimizer.learning_rate.numpy()

        print("Changing learning rate from {0} to {1}".format(old_learning_rate, new_learning_rate))
        optimizer.learning_rate.assign(new_learning_rate)

    for _step, x_batch in enumerate(train_dataset):
        step = _step + steps_pr_epoch * epoch

        # ---- one training step
        res = train_step(model, x_batch, n_samples, optimizer)

        if step % 200 == 0:
            train_history.append(res["elbo"].numpy().mean())

            # ---- write training stats to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('Evaluation/elbo', res["elbo"], step=step)
                tf.summary.scalar('Evaluation/lpxz', res['lpxz'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lqzx', res['lqzx'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/lpz', res['lpz'].numpy().mean(), step=step)
                tf.summary.scalar('Evaluation/kl', res['kl'].numpy().mean(), step=step)

            # ---- collect validation stats
            for x_val_batch in val_dataset:

                val_res = val_step(model, x_val_batch, n_samples)

                val_elbo_metric.update_state(val_res["log_avg_w"])
                val_lpxz_metric.update_state(tf.reduce_mean(val_res['lpxz'], axis=0))
                val_lpz_metric.update_state(tf.reduce_mean(val_res['lpz'], axis=0))
                val_lqzx_metric.update_state(tf.reduce_mean(val_res['lqzx'], axis=0))
                val_kl_metric.update_state(val_res["kl"])

            # ---- summarize the results over the batches and reset the metrics
            val_elbo = val_elbo_metric.result()
            val_elbo_metric.reset_states()
            val_lpxz = val_lpxz_metric.result()
            val_lpxz_metric.reset_states()
            val_lpz = val_lpz_metric.result()
            val_lpz_metric.reset_states()
            val_lqzx = val_lqzx_metric.result()
            val_lqzx_metric.reset_states()
            val_kl = val_kl_metric.result()
            val_kl_metric.reset_states()

            val_history.append(val_elbo)

            # ---- write val stats to tensorboard
            with val_summary_writer.as_default():
                tf.summary.scalar('Evaluation/elbo', val_elbo, step=step)
                tf.summary.scalar('Evaluation/lpxz', val_lpxz, step=step)
                tf.summary.scalar('Evaluation/lqzx', val_lqzx, step=step)
                tf.summary.scalar('Evaluation/lpz', val_lpz, step=step)
                tf.summary.scalar('Evaluation/kl', val_kl, step=step)

            # ---- save the model if the validation loss improves
            if val_elbo > best:
                print("saving model...")
                model.save_weights('/tmp/iwae/task03/best_weights' + '_nsamples_{}'.format(n_samples))
                best = val_elbo

            took = time.time() - start
            start = time.time()

            print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                  .format(epoch, epochs, step, total_steps, res["elbo"].numpy(), val_elbo.numpy(), took))

# ---- plot the training history
plt.clf()
plt.plot(train_history)
plt.plot(val_history)
plt.ylim([-96, -90])
plt.savefig('task03_history' + '_nsamples_{}'.format(n_samples))
plt.close()

# ---- if you want to do early stopping based on the best validation set error:
model.load_weights('/tmp/iwae/task03/best_weights' + '_nsamples_{}'.format(n_samples))

# ---- test-set llh estimate using 5000 samples
test_elbo_metric = utils.MyMetric()
L = 5000

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
utils.plot_prior(model, n=10,
                 epoch=step // steps_pr_epoch,
                 n_latent=n_latent,
                 prefix='task03_',
                 suffix='_nsamples_{}'.format(n_samples))
