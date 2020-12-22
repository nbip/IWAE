import tensorflow as tf
from tensorflow_probability import distributions as tfd


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
                 n_hidden1,
                 n_hidden2,
                 n_latent1,
                 n_latent2,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.encode_x_to_z1 = BasicBlock(n_hidden1, n_latent1)
        self.encode_z1_to_z2 = BasicBlock(n_hidden2, n_latent2)

    def call(self, x, n_samples):

        qz1x = self.encode_x_to_z1(x)

        z1 = qz1x.sample()

        qz2z1 = self.encode_z1_to_z2(z1)

        z2 = qz2z1.sample()

        return z1, qz1x, z2, qz2z1


class Decoder(tf.keras.Model):
    def __init__(self,
                 n_hidden1,
                 n_hidden2,
                 n_latent1,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.decode_z2_to_z1 = BasicBlock(n_hidden2, n_latent1)

        # decode z1 to x
        self.decode_z1_to_x = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh),
                tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh),
                tf.keras.layers.Dense(784, activation=None)
            ]
        )

        # self.l1 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)
        # self.l2 = tf.keras.layers.Dense(n_hidden1, activation=tf.nn.tanh)
        #
        # self.lout = tf.keras.layers.Dense(784, activation=None)

    def call(self, z1, z2):

        pz1z2 = self.decode_z2_to_z1(z2)

        logits = self.decode_z1_to_x(z1)

        pxz1 = tfd.Bernoulli(logits=logits)

        return logits, pxz1, pz1z2


class VAE(tf.keras.Model):
    def __init__(self,
                 n_hidden1,
                 n_hidden2,
                 n_latent1,
                 n_latent2,
                 **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.encoder = Encoder(n_hidden1, n_hidden2, n_latent1, n_latent2)
        self.decoder = Decoder(n_hidden1, n_hidden2, n_latent1)

    def call(self, x, n_samples, beta=1.0):

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

        # ---- kl divergences
        kl1 = tf.reduce_sum(tfd.kl_divergence(qz1x, pz1z2), axis=-1)

        kl2 = tf.reduce_sum(tfd.kl_divergence(qz2z1, pz2), axis=-1)

        # kl2_check = - 0.5 * (1 + tf.math.log(qz2z1.scale ** 2) - qz2z1.loc ** 2 - qz2z1.scale ** 2 )

        vae_elbo = tf.reduce_mean(lpxz1 - kl1 - kl2, axis=-1)
        beta_vae_elbo = tf.reduce_mean(lpxz1 - beta * kl1 - beta * kl2, axis=-1)

        # log weights
        log_w = lpxz1 + lpz1z2 + lpz2 - lqz1x - lqz2z1

        # loss comparison
        iwae_elbo = tf.reduce_mean(log_w, axis=-1)
        beta_iwae_elbo = tf.reduce_mean(lpxz1 + beta * (lpz1z2 + lpz2) - lqz1x - lqz2z1, axis=-1) +\
                         tf.reduce_mean((1 - beta) * (lqz1x + lqz2z1), axis=-1)

        # mean over minibatch
        loss = -beta_vae_elbo

        return {"loss": loss,
                "vae_elbo": vae_elbo,
                "beta_vae_elbo": beta_vae_elbo,
                "iwae_elbo": iwae_elbo,
                "beta_iwae_elbo": beta_iwae_elbo,
                "kl1": kl1,
                "kl2": kl2,
                "z1": z1,
                "z2": z2,
                "logits": logits,
                "lpxz1": lpxz1,
                "lpz1z2": lpz1z2,
                "lpz2": lpz2,
                "lqz1x": lqz1x,
                "lqz2z1": lqz2z1}
