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
sys.path.insert(0, './tasks')
import utils
import iwae1
import iwae2
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt
import cycler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---- plot settings
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 15.0
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['figure.autolayout'] = True
color = plt.cm.viridis(np.linspace(0, 1, 10))
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

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
string = "task04_{0}_{1}_{2}".format(args.objective, args.stochastic_layers, args.n_samples)

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


class CIWAE(iwae1.IWAE):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(CIWAE, self).__init__(n_hidden, n_latent, **kwargs)

        self.conditional_prior_network = iwae1.BasicBlock(n_hidden, n_latent, **kwargs)

    def call(self, x, y, n_samples, beta=1.0):

        y_onehot = tf.one_hot(tf.cast(y, tf.uint8), depth=10)

        # ---- encode
        xy = tf.concat([x, y_onehot], axis=-1)
        z, qzxy  = self.encoder(xy, n_samples)

        # ---- decode
        zy = tf.concat([z, tf.tile(y_onehot[None, :, :], [n_samples, 1, 1])], axis=-1)
        logits, pxzy = self.decoder(zy)

        # ---- conditional prior
        pzy = self.conditional_prior_network(y_onehot)

        # ---- the prior does not have to be conditional on y
        # pz = tfd.Normal(0, 1)

        # ---- loss
        lpzy = tf.reduce_sum(pzy.log_prob(z), axis=-1)

        lqzxy = tf.reduce_sum(qzxy.log_prob(z), axis=-1)

        lpxzy = tf.reduce_sum(pxzy.log_prob(x), axis=-1)

        log_w = lpxzy + beta * (lpzy - lqzxy)

        # ---- regular VAE elbos
        kl = tf.reduce_sum(tfd.kl_divergence(qzxy, pz), axis=-1)
        kl2 = -tf.reduce_mean(lpzy - lqzxy, axis=0)

        # mean over samples and batch
        vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
        vae_elbo_kl = tf.reduce_mean(lpxzy) - beta * tf.reduce_mean(kl)

        # ---- IWAE elbos
        # eq (8): logmeanexp over samples and mean over batch
        iwae_elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)

        # eq (14):
        m = tf.reduce_max(log_w, axis=0, keepdims=True)
        log_w_minus_max = log_w - m
        w = tf.exp(log_w_minus_max)
        w_normalized = w / tf.reduce_sum(w, axis=0, keepdims=True)
        w_normalized_stopped = tf.stop_gradient(w_normalized)

        iwae_eq14 = tf.reduce_mean(tf.reduce_sum(w_normalized_stopped * log_w, axis=0))

        # ---- self-normalized importance sampling
        al = tf.nn.softmax(log_w, axis=0)

        snis_z = tf.reduce_sum(al[:, :, None] * z, axis=0)

        return {"vae_elbo": vae_elbo,
                "vae_elbo_kl": vae_elbo_kl,
                "iwae_elbo": iwae_elbo,
                "iwae_eq14": iwae_eq14,
                "z": z,
                "snis_z": snis_z,
                "al": al,
                "logits": logits,
                "lpxzy": lpxzy,
                "lpzy": lpzy,
                "lqzxy": lqzxy}

    @tf.function
    def train_step(self, x, y, n_samples, beta, optimizer, objective="vae_elbo"):
        with tf.GradientTape() as tape:
            res = self.call(x, y, n_samples, beta)
            loss = -res[objective]

        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return res

    @tf.function
    def val_step(self, x, y, n_samples, beta):
        return self.call(x, y, n_samples, beta)

    def sample(self, z, y):

        y_onehot = tf.one_hot(tf.repeat(y, z.shape[0]), depth=10)

        pzy = self.conditional_prior_network(y_onehot)

        # z = pzy.sample()
        z_new = pzy.loc + pzy.scale * z

        zy = tf.concat([z_new, y_onehot], axis=-1)

        logits = self.decoder.decode_z_to_x(zy)

        probs = tf.nn.sigmoid(logits)

        pxz = tfd.Bernoulli(logits=logits)

        x_sample = pxz.sample()

        return x_sample, probs

    def generate_and_save_images(self, z, epoch, string):

        n = 10

        canvas = np.zeros((n * 28, 2 * n * 28))

        for i in range(10):
            # ---- samples from the prior, conditional on the label
            x_samples, x_probs = self.sample(z, i)
            x_samples = x_samples.numpy().squeeze()
            x_probs = x_probs.numpy().squeeze()

            for j in range(n):
                canvas[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_samples[j].reshape(28, 28)
                canvas[i * 28: (i + 1) * 28, n * 28 + j * 28: n * 28 + (j + 1) * 28] = x_probs[j].reshape(28, 28)
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.imshow(canvas, cmap='gray_r')
        plt.title("epoch {:04d}".format(epoch), fontsize=50)
        plt.axis('off')
        plt.savefig('./results/' + string + '_image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def generate_and_save_posteriors(self, x, y, n_samples, epoch, string):

        # ---- posterior snis means
        res = self.call(x, y, n_samples)

        snis_z = res["snis_z"]

        # pca
        scaler = StandardScaler()
        pca = PCA(n_components=2)

        snis_z = scaler.fit_transform(snis_z)
        pca.fit(snis_z)
        z = pca.transform(snis_z)

        plt.clf()
        for c in np.unique(y):
            plt.scatter(z[y == c, 0], z[y == c, 1], s=10, label=str(c))
        plt.legend(loc=(1.04,0))
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.title("epoch {:04d}".format(epoch), fontsize=50)
        plt.savefig('./results/' + string + '_posterior_at_epoch_{:04d}.png'.format(epoch))
        plt.close()


    @staticmethod
    def write_to_tensorboard(res, step):
        tf.summary.scalar('Evaluation/vae_elbo', res["vae_elbo"], step=step)
        tf.summary.scalar('Evaluation/iwae_elbo', res["iwae_elbo"], step=step)
        tf.summary.scalar('Evaluation/lpxzy', res['lpxzy'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lqzxy', res['lqzxy'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lpzy', res['lpzy'].numpy().mean(), step=step)


# ---- instantiate the model, optimizer and metrics
if args.stochastic_layers == 1:
    n_latent = [100]
    n_hidden = [200]
    model = CIWAE(n_hidden[0], n_latent[0])
else:
    n_latent = [100, 50]
    n_hidden = [200, 100]
    model = iwae2.IWAE(n_hidden, n_latent)

optimizer = keras.optimizers.Adam(learning_rate_dict[0], epsilon=1e-4)
print("Initial learning rate: ", optimizer.learning_rate.numpy())

# ---- prepare plotting of samples during training
# use the same samples from the prior throughout training
pz = tfd.Normal(0, 1)
z = pz.sample([10, n_latent[-1]])

plt_epochs = list(2**np.arange(12))
plt_epochs.insert(0, 0)
plt_epochs.append(epochs-1)

# ---- binarize the test data
# we'll only do this once, while the training data is binarized at the
# start of each epoch
Xtest = utils.bernoullisample(Xtest)
model.generate_and_save_posteriors(Xtest, ytest, 10, 0, string)

# ---- do the training
start = time.time()
best = float(-np.inf)

for epoch in range(epochs):

    # ---- binarize the training data at the start of each epoch
    Xtrain_binarized = utils.bernoullisample(Xtrain)

    train_data = tf.data.Dataset.from_tensor_slices(Xtrain_binarized)
    train_labels = tf.data.Dataset.from_tensor_slices(ytrain.astype(np.float32))
    train_dataset = (tf.data.Dataset.zip((train_data, train_labels))
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

    for _step, (x_batch, y_batch) in enumerate(train_dataset):
        step = _step + steps_pr_epoch * epoch

        # ---- warm-up
        beta = 1.0
        # beta = np.min([step / 200000, 1.0]).astype(np.float32)

        # ---- one training step
        res = model.train_step(x_batch, y_batch, n_samples, beta, optimizer, objective=objective)

        if step % 200 == 0:

            # ---- write training stats to tensorboard
            with train_summary_writer.as_default():
                model.write_to_tensorboard(res, step)

            # ---- monitor the test-set
            test_res = model.val_step(Xtest, ytest, n_samples, beta)

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
for i, (x, y) in enumerate(zip(Xtest, ytest)):
    res = model(x[None, :].astype(np.float32), y[None], L)
    test_elbo_metric.update_state(res['iwae_elbo'][None, None])
    if i % 200 == 0:
        print("{0}/{1}".format(i, Ntest))

test_set_llh = test_elbo_metric.result()
test_elbo_metric.reset_states()

print("Test-set {0} sample log likelihood estimate: {1:.4f}".format(L, test_set_llh))
