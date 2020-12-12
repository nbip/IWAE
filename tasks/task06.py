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
import utils


def logmeanexp(log_w):
    max = tf.reduce_max(log_w, axis=0)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=0)) + max


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

# ---- load data
(Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

# ---- reshape to vectors
Xtrain = Xtrain.reshape(Ntrain, -1) / 255
Xtest = Xtest.reshape(Ntest, -1) / 255


# TODO: define a basis block as in https://github.com/xqding/Importance_Weighted_Autoencoders/blob/master/model/vae_models.py
class BasisBlock(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(BasisBlock, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)
        self.lmu = tf.keras.layers.Dense(n_latent1, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent1, activation=tf.exp)

    def call(self, input, n_samples):

        h1 = self.l1(input)
        h2 = self.l2(h1)
        q_mu = self.lmu1(h2)
        q_std = self.lstd1(h2)

        qz_input = tfd.Normal(q_mu, q_std + 1e-6)

        z = qz_input.sample(n_samples)

        return z, qz_input


class Encoder(tf.keras.Model):
    def __init__(self,
                 n_hidden1,
                 n_hidden2,
                 n_latent1,
                 n_latent2,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)
        self.lmu1 = tf.keras.layers.Dense(n_latent1, activation=None)
        self.lstd1 = tf.keras.layers.Dense(n_latent1, activation=tf.exp)

        self.l3 = tf.keras.layers.Dense(n_hidden2, activation=tf.nn.tanh)
        self.l4 = tf.keras.layers.Dense(n_hidden2, activation=tf.nn.tanh)
        self.lmu2 = tf.keras.layers.Dense(n_latent2, activation=None)
        self.lstd2 = tf.keras.layers.Dense(n_latent2, activation=tf.exp)

        # self.encoder_z1 = BasisBlock(n_hidden1, n_latent1)
        # self.encoder_z2 = BasisBlock(n_hidden2, n_latent2)

    def call(self, x, n_samples):

        h1 = self.l1(x)
        h2 = self.l2(h1)
        q_mu1 = self.lmu1(h2)
        q_std1 = self.lstd1(h2)

        qz1x = tfd.Normal(q_mu1, q_std1 + 1e-6)

        z1 = qz1x.sample(n_samples)

        h3 = self.l3(z1)
        h4 = self.l4(h3)
        q_mu2 = self.lmu2(h4)
        q_std2 = self.lstd2(h4)

        qz2z1 = tfd.Normal(q_mu2, q_std2 + 1e-6)

        z2 = qz2z1.sample()

        return z1, qz1x, z2, qz2z1


class Decoder(tf.keras.Model):
    def __init__(self,
                 n_hidden1,
                 n_hidden2,
                 n_latent1,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)

        self.lmu = tf.keras.layers.Dense(n_latent1, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent1, activation=tf.exp)

        self.l3 = tf.keras.layers.Dense(n_hidden2, activation=tf.nn.tanh)
        self.l4 = tf.keras.layers.Dense(n_hidden2, activation=tf.nn.tanh)

        self.lout = tf.keras.layers.Dense(784, activation=None)

    def call(self, z1, z2):

        h4 = self.l4(z2)
        h3 = self.l3(h4)

        mu1 = self.lmu(h3)
        std1 = self.lstd(h3)

        pz1z2 = tfd.Normal(mu1, std1 + 1e-6)

        h2 = self.l2(z1)
        h1 = self.l1(h2)
        logits = self.lout(h1)

        pxz1 = tfd.Bernoulli(logits=logits)

        return logits, pxz1, pz1z2


class IWAE(tf.keras.Model):
    def __init__(self,
                 n_hidden1,
                 n_hidden2,
                 n_latent1,
                 n_latent2,
                 **kwargs):
        super(IWAE, self).__init__(**kwargs)

        self.encoder = Encoder(n_hidden1, n_hidden2, n_latent1, n_latent2)
        self.decoder = Decoder(n_hidden1, n_hidden2, n_latent1)

    def call(self, x, n_samples):

        # ---- encode/decode
        z1, qz1x, z2, qz2z1 = self.encoder(x, n_samples)

        logits, pxz1, pz1z2 = self.decoder(z1, z2)

        # ---- loss
        pz2 = tfd.Normal(0, 1)

        lpz2 = tf.reduce_sum(pz2.log_prob(z2), axis=-1)

        lqz2z1 = tf.reduce_sum(qz2z1.log_prob(z2), axis=-1)

        lpz1z2 = tf.reduce_sum(pz1z2.log_prob(z1), axis=-1)

        lqz1x = tf.reduce_sum(qz1x.log_prob(z1), axis=-1)

        lpxz1 = tf.reduce_sum(pxz1.log_prob(x), axis=-1)

        # kl_qzx_pz = tf.reduce_sum(tfd.kl_divergence(qzx, pz), axis=-1)

        # vae_elbo = -kl_qzx_pz + tf.reduce_mean(lpxz)

        # log weights
        log_w = lpxz1 + lpz1z2 + lpz2 - lqz1x - lqz2z1

        # average over samples
        log_avg_w = logmeanexp(log_w)

        # average over batch
        elbo = tf.reduce_mean(log_avg_w, axis=-1)

        # loss is the negative elbo
        loss = -elbo

        return {"loss": loss,
                "elbo": elbo,
                # "kl": kl_qzx_pz,
                # "vae_elbo": vae_elbo,
                "log_avg_w": log_avg_w,
                "z1": z1,
                "z2": z2,
                "logits": logits,
                "lpxz1": lpxz1,
                "lpz1z2": lpz1z2,
                "lpz2": lpz2,
                "lqz1x": lqz1x,
                "lqz2z1": lqz2z1}


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
train_log_dir = "/tmp/iwae/task06/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# ---- prepare the data
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

steps_pr_epoch = Ntrain // batch_size
total_steps = steps_pr_epoch * epochs

# ---- instantiate the model, optimizer and metrics
model = IWAE(n_hidden1,
             n_hidden2,
             n_latent1,
             n_latent2)

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

            took = time.time() - start
            start = time.time()

            print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, time: {5:.2f}"
                  .format(epoch, epochs, step, total_steps, res["elbo"].numpy(), took))

# ---- save final weights
model.save_weights('/tmp/iwae/task06/final_weights' + '_nsamples_{}'.format(n_samples))

# ---- plot the training history
plt.clf()
plt.plot(train_history)
plt.ylim([-96, -80])
plt.savefig('task06_history' + '_nsamples_{}'.format(n_samples))
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
