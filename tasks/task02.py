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


parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=5, help="number of importance samples")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--epochs", type=int, default=-1,
                    help="numper of epochs, if set to -1 number of epochs "
                         "will be set based on the learning rate scheme from the paper")
parser.add_argument("--gpu", type=str, default='0', help="Choose GPU")
args = parser.parse_args()
print(args)


# --- override the call method in iwae1
class IWAEDReG(iwae1.IWAE):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(IWAEDReG, self).__init__(n_hidden, n_latent, **kwargs)

    def call(self, x, n_samples, beta=1.0):
        # ---- encode/decode
        z, qzx  = self.encoder(x, n_samples)

        logits, pxz = self.decoder(z)

        # ---- loss
        pz = tfd.Normal(0, 1)

        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)

        log_w = lpxz + beta * (lpz - lqzx)

        # ---- IWAE elbo
        # eq (8): logmeanexp over samples and mean over batch
        iwae_elbo = tf.reduce_mean(utils.logmeanexp(log_w, axis=0), axis=-1)

        # ---- self-normalized importance sampling
        al = tf.nn.softmax(log_w, axis=0)

        snis_z = tf.reduce_sum(al[:, :, None] * z, axis=0)

        # ---- prepare DReG
        q_mu_stopped, q_std_stopped = tf.stop_gradient(qzx.loc), tf.stop_gradient(qzx.scale)

        qzx_stopped = tfd.Normal(q_mu_stopped, q_std_stopped + 1e-6)

        lqzx_stopped = tf.reduce_sum(qzx_stopped.log_prob(z), axis=-1)

        stopped_normalized_weights = tf.stop_gradient(al)
        sq_normalized_weights = tf.square(stopped_normalized_weights)

        stopped_log_w = lpz + lpxz - lqzx_stopped

        # ---- doubly reparameterized
        DReG = tf.reduce_sum(sq_normalized_weights * stopped_log_w, axis=0)

        # ---- average over minibatch to get the average DReG
        inference_loss = -tf.reduce_mean(DReG, axis=-1)

        return {"iwae_elbo": iwae_elbo,
                "inference_loss": inference_loss,
                "z": z,
                "snis_z": snis_z,
                "logits": logits,
                "lpxz": lpxz,
                "lpz": lpz,
                "lqzx": lqzx}

    @tf.function
    def train_step(self, x, n_samples, beta, optimizer):

        with tf.GradientTape(persistent=True) as tape:
            res = self.call(x, n_samples, beta)
            loss = -res["iwae_elbo"]
            inference_loss = res["inference_loss"]

        encoder_grads = tape.gradient(inference_loss, self.encoder.trainable_weights)
        decoder_grads = tape.gradient(loss, self.decoder.trainable_weights)
        del tape
        optimizer.apply_gradients(zip(encoder_grads + decoder_grads,
                                      self.encoder.trainable_weights + self.decoder.trainable_weights))

        return res

    @staticmethod
    def write_to_tensorboard(res, step):
        tf.summary.scalar('Evaluation/iwae_elbo', res["iwae_elbo"], step=step)
        tf.summary.scalar('Evaluation/lpxz', res['lpxz'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lqzx', res['lqzx'].numpy().mean(), step=step)
        tf.summary.scalar('Evaluation/lpz', res['lpz'].numpy().mean(), step=step)


# ---- string describing the experiment, to use in tensorboard and plots
string = "task02_{0}".format(args.n_samples)

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
n_latent = [100]
n_hidden = [200]
model = IWAEDReG(n_hidden[0], n_latent[0])

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
        res = model.train_step(x_batch, n_samples, beta, optimizer)

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
                  .format(epoch, epochs, step, total_steps, res["iwae_elbo"].numpy(), test_res["iwae_elbo"], took))

# ---- save final weights
model.save_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- load the final weights?
# model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- test-set llh estimate using 5000 samples
test_elbo_metric = utils.MyMetric()
L = 5000

# ---- since we are using 5000 importance samples we have to loop over each element of the test-set
for i, x in enumerate(Xtest):
    res = model.call(x[None, :].astype(np.float32), L)
    test_elbo_metric.update_state(res['iwae_elbo'][None, None])
    if i % 200 == 0:
        print("{0}/{1}".format(i, Ntest))

test_set_llh = test_elbo_metric.result()
test_elbo_metric.reset_states()

print("Test-set {0} sample log likelihood estimate: {1:.4f}".format(L, test_set_llh))
