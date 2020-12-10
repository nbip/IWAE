import tensorflow as tf
from tensorflow_probability import distributions as tfd


def logmeanexp(log_w):
    max = tf.reduce_max(log_w, axis=0)
    return tf.math.log(tf.reduce_mean(tf.exp(log_w - max), axis=0)) + max


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

        kl_qzx_pz = tf.reduce_sum(tfd.kl_divergence(qzx, pz), axis=-1)

        vae_elbo = -kl_qzx_pz + tf.reduce_mean(lpxz)

        # log weights
        log_w = lpxz + lpz - lqzx

        # average over samples
        log_avg_w = logmeanexp(log_w)

        # average over batch
        elbo = tf.reduce_mean(log_avg_w, axis=-1)

        # loss is the negative elbo
        loss = -elbo

        return {"loss": loss,
                "elbo": elbo,
                "kl": kl_qzx_pz,
                "vae_elbo": vae_elbo,
                "log_avg_w": log_avg_w,
                "z": z,
                "logits": logits,
                "lpxz": lpxz,
                "lpz": lpz,
                "lqzx": lqzx}
