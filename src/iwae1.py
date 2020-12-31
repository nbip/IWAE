import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import matplotlib
matplotlib.use('Agg')  # needed when running from commandline
import matplotlib.pyplot as plt


# ---- plot settings
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 15.0
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['figure.autolayout'] = True


def logmeanexp(log_w, axis):
    max = tf.reduce_max(log_w, axis=axis)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=axis)) + max


class BasicBlock(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.lmu = tf.keras.layers.Dense(n_latent, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent, activation=tf.exp)

    def call(self, input):
        h1 = self.l1(input)
        h2 = self.l2(h1)
        q_mu = self.lmu(h2)
        q_std = self.lstd(h2)

        qz_given_input = tfd.Normal(q_mu, q_std + 1e-6)

        return qz_given_input


class Encoder(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.encode_x_to_z = BasicBlock(n_hidden, n_latent)

    def call(self, x, n_samples):
        qzx = self.encode_x_to_z(x)

        z = qzx.sample(n_samples)

        return z, qzx


class Decoder(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.decode_z_to_x = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh),
                tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh),
                tf.keras.layers.Dense(784, activation=None)
            ]
        )

    def call(self, z):

        logits = self.decode_z_to_x(z)

        pxz = tfd.Bernoulli(logits=logits)

        return logits, pxz


class IWAE(tf.keras.Model):
    def __init__(self,
                 n_hidden,
                 n_latent,
                 **kwargs):
        super(IWAE, self).__init__(**kwargs)

        self.encoder = Encoder(n_hidden, n_latent)
        self.decoder = Decoder(n_hidden)

    def call(self, x, n_samples, beta=1.0):
        # ---- encode/decode
        z, qzx = self.encoder(x, n_samples)

        logits, pxz = self.decoder(z)

        # ---- loss
        pz = tfd.Normal(0, 1)

        lpz = tf.reduce_sum(pz.log_prob(z), axis=-1)

        lqzx = tf.reduce_sum(qzx.log_prob(z), axis=-1)

        lpxz = tf.reduce_sum(pxz.log_prob(x), axis=-1)

        log_w = lpxz + lpz - lqzx

        kl = tf.reduce_sum(tfd.kl_divergence(qzx, pz), axis=-1)
        kl2 = -tf.reduce_mean(lpz - lqzx, axis=0)

        # mean over samples and batch
        vae_elbo = tf.reduce_mean(tf.reduce_mean(log_w, axis=0), axis=-1)
        vae_elbo_kl = tf.reduce_mean(lpxz) - tf.reduce_mean(kl)

        # logmeanexp over samples and mean over batch
        iwae_elbo = tf.reduce_mean(logmeanexp(log_w, axis=0), axis=-1)

        m = tf.reduce_max(log_w, axis=0, keepdims=True)
        log_w_minus_max = log_w - m
        w = tf.exp(log_w_minus_max)
        w_normalized = w / tf.reduce_sum(w, axis=0, keepdims=True)
        w_normalized_stopped = tf.stop_gradient(w_normalized)

        iwae_eq14 = tf.reduce_mean(tf.reduce_sum(w_normalized_stopped * log_w, axis=0))

        return {"vae_elbo": vae_elbo,
                "iwae_elbo": iwae_elbo,
                "iwae_eq14": iwae_eq14,
                "vae_elbo_kl": vae_elbo_kl,
                "z": z,
                "logits": logits,
                "lpxz": lpxz,
                "lpz": lpz,
                "lqzx": lqzx}

    @tf.function
    def train_step(self, x, n_samples, beta, optimizer, objective="vae_elbo"):
        with tf.GradientTape() as tape:
            res = self.call(x, n_samples, beta)
            loss = -res[objective]

        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return res

    @tf.function
    def val_step(self, x, n_samples, beta):
        return self.call(x, n_samples, beta)

    def sample(self, z):

        logits = self.decoder.decode_z_to_x(z)

        probs = tf.nn.sigmoid(logits)

        pxz = tfd.Bernoulli(logits=logits)

        x_sample = pxz.sample()

        return x_sample, probs

    def generate_and_save_images(self, z, epoch, string):

        x_samples, x_probs = self.sample(z)
        x_samples = x_samples.numpy().squeeze()
        x_probs = x_probs.numpy().squeeze()

        n = int(np.sqrt(x_samples.shape[0]))

        canvas = np.zeros((n * 28, 2 * n * 28))

        for i in range(n):
            for j in range(n):
                canvas[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_samples[i * n + j].reshape(28, 28)
                canvas[i * 28: (i + 1) * 28, n * 28 + j * 28: n * 28 + (j + 1) * 28] = x_probs[i * n + j].reshape(28,
                                                                                                                  28)

        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.imshow(canvas, cmap='gray_r')
        plt.title("epoch {:04d}".format(epoch))
        plt.axis('off')
        plt.savefig(string + 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()



