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
sys.path.insert(0, './src')
sys.path.insert(0, './notebooks/IWAE/src')
import utils
import model


def logmeanexp(log_w):
    max = tf.reduce_max(log_w, axis=0)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=0)) + max

# ---- set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
epochs = 0
learning_rate_change_epoch = []
learning_rates = []

for i in range(8):
    learning_rates.append(0.001 * 10**(-i/7))
    epochs += 3 ** i
    learning_rate_change_epoch.append(epochs)

# ---- experiment settings
n_latent = 100
n_samples = 5
n_hidden = 200
batch_size = 20

# ---- load data
(Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

# ---- reshape to vectors
Xtrain = Xtrain.reshape(Ntrain, -1) / 255
Xtest = Xtest.reshape(Ntest, -1) / 255

class Encoder(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.lmu = tf.keras.layers.Dense(n_latent, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent, activation=tf.exp)

    def call(self, x, n_samples):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        q_mu = self.lmu(h2)
        q_std = self.lstd(h2)

        qzx = tfd.Normal(q_mu, q_std + 1e-6)

        z = qzx.sample(n_samples)

        return z, qzx


class Decoder(tf.keras.Model):
    def __init__(self, n_hidden, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.lout = tf.keras.layers.Dense(784, activation=None)

    def call(self, z):

        h1 = self.l1(z)
        h2 = self.l2(h1)
        logits = self.lout(h2)

        pxz = tfd.Bernoulli(logits=logits)

        return logits, pxz


class VAE(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.encoder = Encoder(n_hidden, n_latent)
        self.decoder = Decoder(n_hidden)

    def call(self, x, n_samples):

        # ---- encode/decode
        z, qzx = self.encoder(x, n_samples)

        logits, pxz = self.decoder(z)

        # ---- loss
        pz = tfd.Normal(0, 1)

        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)

        # log weights
        log_w = lpxz + lpz - lqzx

        # --- different versions of the VAE and IWAE losses

        # ---- plain VAE, SGVB A
        vae_a = tf.reduce_mean(log_w)

        # ---- plain VAE, SGVB B
        kl_qzx_pz = tf.reduce_sum(tfd.kl_divergence(qzx, pz), axis=-1)
        vae_b = tf.reduce_mean(-kl_qzx_pz + lpxz)

        # ---- IWAE eq 8
        log_avg_w = logmeanexp(log_w)
        iwae_eq8 = tf.reduce_mean(log_avg_w, axis=-1)

        # ---- IWAE eq 8
        log_avg_w = tf.reduce_logsumexp(log_w, axis=0) - tf.math.log(tf.cast(tf.constant(n_samples), tf.float32))
        iwae_eq82 = tf.reduce_mean(log_avg_w, axis=-1)

        # ---- IWAE eq 14
        log_w_stopped = tf.stop_gradient(log_w)
        normalized_w = tf.nn.softmax(log_w_stopped, axis=0)
        objective = tf.reduce_sum(normalized_w * log_w, axis=0)
        iwae_eq14 = tf.reduce_mean(objective, axis=-1)

        # ---- IWAE eq 14
        log_w_stopped = tf.stop_gradient(log_w)
        log_w_minus_max = log_w_stopped - tf.reduce_max(log_w_stopped, keepdims=True, axis=0)
        w = tf.exp(log_w_minus_max)
        normalized_w = w / tf.reduce_sum(w, keepdims=True, axis=0)
        objective = tf.reduce_sum(normalized_w * log_w, axis=0)
        iwae_eq142 = tf.reduce_mean(objective, axis=-1)

        # ---- IWAE eq 14
        log_w_stopped = tf.stop_gradient(log_w)
        max = tf.reduce_max(log_w_stopped, axis=0)
        w_minus_max = tf.exp(log_w_stopped - max)
        objective = tf.math.log(tf.reduce_mean(w_minus_max, axis=0)) + max
        iwae_eq143 = tf.reduce_mean(objective, axis=-1)

        return {"vae_a": vae_a,
                "vae_b": vae_b,
                "iwae_eq8": iwae_eq8,
                "iwae_eq82": iwae_eq82,
                "iwae_eq14": iwae_eq14,
                "iwae_eq142": iwae_eq142,
                "iwae_eq143": iwae_eq143,
                "kl": kl_qzx_pz,
                "log_avg_w": log_avg_w,
                "z": z,
                "logits": logits,
                "lpxz": lpxz,
                "lpz": lpz,
                "lqzx": lqzx}


# ---- instantiate the model, optimizer and metrics
model = VAE(n_hidden, n_latent)

# ---- binarize the training data at the start of each epoch
Xtrain_binarized = utils.bernoullisample(Xtrain)

train_dataset = (tf.data.Dataset.from_tensor_slices(Xtrain_binarized)
    .shuffle(Ntrain).batch(batch_size))

for _step, x_batch in enumerate(train_dataset):
    break


res = model(x_batch, 1)

print(res["vae_a"],
      res["vae_b"],
      res["iwae_eq8"],
      res["iwae_eq82"],
      res["iwae_eq14"],
      res["iwae_eq142"],
      res["iwae_eq143"])

# ---- train step
@tf.function
def train_step(model, x, n_samples, optimizer):

    with tf.GradientTape() as tape:
        res = model(x, n_samples)
        loss = res["loss"]

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return res


# ---- prepare tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "/tmp/iwae/task10/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# ---- prepare the data
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

steps_pr_epoch = Ntrain // batch_size
total_steps = steps_pr_epoch * epochs

# ---- instantiate the model, optimizer and metrics
model = model.VAE(n_hidden, n_latent)

optimizer = keras.optimizers.Adam(learning_rates[0])
print("Initial learning rate: ", optimizer.learning_rate.numpy())

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
    if args.epochs == -1 and np.sum(epoch == np.asarray(learning_rate_change_epoch)) > 0:  # TODO: if epoch in.. ?
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

            took = time.time() - start
            start = time.time()

            print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, time: {5:.2f}"
                  .format(epoch, epochs, step, total_steps, res["elbo"].numpy(), took))

# ---- save final weights
model.save_weights('/tmp/iwae/task10/final_weights' + '_nsamples_{}'.format(n_samples))

# ---- plot the training history
plt.clf()
plt.plot(train_history)
plt.ylim([-96, -80])
plt.savefig('task10_history' + '_nsamples_{}'.format(n_samples))
plt.close()

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
utils.plot_prior(model, n=10,
                 epoch=step // steps_pr_epoch,
                 n_latent=n_latent,
                 prefix='task10_',
                 suffix='_nsamples_{}'.format(n_samples))
