import tensorflow as tf
from tensorflow.keras import layers
import os
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
import glob
import random
import warnings
import pickle
from sklearn import preprocessing


normalizer = preprocessing.MaxAbsScaler()
audio_dir = os.path.join(os.getcwd(), 'music-data')
#filenames = glob.glob(audio_dir + '/fma_small/' + '/*[0-9]/*')

with open('electronic-songs.p', 'rb') as f:
    filenames = pickle.load(f)

random.seed(8)
random.shuffle(filenames)

SR = 14400
INPUT_LENGTH = 20
BATCH_SIZE = 16
SOUND_DIM = 844
EPOCHS = 50
LATENT_DIM = 6
restore_latest = False


# https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
def load_mp3s():
    for i in range(len(filenames)):
        # ignore PySoundFile failure warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                x, _ = librosa.load(filenames[i], sr=SR, res_type='kaiser_fast')
            except:
                continue

        # random offset / padding
        if len(x) > INPUT_LENGTH:
            max_offset = len(x) - INPUT_LENGTH
            offset = np.random.randint(max_offset)
            x = x[offset:(INPUT_LENGTH+offset)]
        else:
            if INPUT_LENGTH > len(x):
                max_offset = INPUT_LENGTH - len(x)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            x = np.pad(x, (offset, INPUT_LENGTH - len(x) - offset), 'constant')
        
        # audio data is loaded in the range [-1, 1]
        yield x.reshape(-1, 1) # + pos_encoding ?


def load_spectrograms():
    for i in range(len(filenames)):
        # ignore PySoundFile failure warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                x, _ = librosa.load(filenames[i], sr=SR, res_type='kaiser_fast')
            except:
                continue

        spec = librosa.feature.mfcc(y=x, sr=SR)
        spec = normalizer.fit_transform(spec)  # normalize in range [-1, 1]

        if spec.shape[-1] < SOUND_DIM:
            offset = SOUND_DIM - spec.shape[-1]
            spec = np.pad(spec, ((0, 0), (0, offset)), 'constant')
        elif spec.shape[-1] > SOUND_DIM:
            spec = spec[:, :SOUND_DIM]
        yield spec


@tf.function
def scaled_dot_product_attention(q, k, v):
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    z = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
    attn_weights = tf.nn.softmax(z)
    output = tf.matmul(tf.nn.softmax(z, axis=-1), v)
    return output, attn_weights


class MultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttn, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.linear = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        # split last dimension into (num_heads, depth)
        # transpose result to shape (batch_size, num_heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attn, attn_weights = scaled_dot_product_attention(q, k, v)
        
        scaled_attn = tf.transpose(scaled_attn, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)
        concat_attn = tf.reshape(scaled_attn, (batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)
        
        output = self.linear(concat_attn) # (batch_size, seq_len_q, d_model)
        
        return output, attn_weights

    
class Generator(tf.keras.Model):
    def __init__(self, input_length, d_model):
        super(Generator, self).__init__()
        
        self.seq_len = input_length
        self.d_model = d_model
        
        self.leaky_relu = layers.LeakyReLU()
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dense1 = layers.Dense(self.seq_len * d_model)
        self.mha1 = MultiHeadAttn(d_model, 4)
        self.mha2 = MultiHeadAttn(d_model, 4)
    
        self.linear = layers.Dense(self.seq_len)
        self.fake_output = layers.Dense(d_model, activation='tanh')
        
    def call(self, x):
        x = self.dense1(x)
        x = self.layer_norm1(x)
        x = self.leaky_relu(x)
        
        x = tf.reshape(x, [-1, self.seq_len, self.d_model])

        # self attention blocks
        x, _ = self.mha1(x, x, x)
        x = self.layer_norm2(x)
        x = self.leaky_relu(x)
        
        x, _ = self.mha2(x, x, x)
        x = self.layer_norm3(x)
        x = self.leaky_relu(x)
        
        x = self.linear(x)
        return self.fake_output(x)


class Discriminator(tf.keras.Model):
    def __init__(self, input_length, d_model, rate=0.1):
        super(Discriminator, self).__init__()
        
        self.seq_len = input_length
        self.d_model = d_model
        
        self.leaky_relu = layers.LeakyReLU()
        self.flatten = layers.Flatten()
        
        self.mha1 = MultiHeadAttn(d_model, 4)
        self.mha2 = MultiHeadAttn(d_model, 4)

        self.linear = layers.Dense(d_model)
        self.out = layers.Dense(1)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
        
    def call(self, x, training=True):
        attn1, _ = self.mha1(x, x, x)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1)
        out1 = self.leaky_relu(out1)

        attn2, _ = self.mha2(out1, out1, out1)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2)
        out2 = self.leaky_relu(out2)

        out3 = self.linear(out2)
        out3 = self.dropout3(out3, training=training)
        out3 = self.layernorm3(out3)
        out3 = self.leaky_relu(out3)

        prediction = self.flatten(out3)
        return self.out(prediction)


class MusicGAN:
    def __init__(self, input_length, d_model, dropout=0.1, d_noise=0.05, g_lr=1e-4, d_lr=1e-5, train=False):
        self.input_length = input_length
        self.d_model = d_model
        self.d_noise = d_noise

        self.sgd_rate = 32
        self.gen_opt = tf.keras.optimizers.Adam(g_lr)
        self.dis_opt = tf.keras.optimizers.Adam(d_lr)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.d_metric_loss = tf.keras.metrics.BinaryCrossentropy(name='dis_loss')
        self.g_metric_loss = tf.keras.metrics.BinaryCrossentropy(name='gen_loss')

        self.generator = Generator(input_length, d_model)
        self.discriminator = Discriminator(input_length, d_model, rate=dropout)

        if train:
            self.g_accumulators = [tf.Variable(tf.zeros_like(tv.initialized_value(), trainable=False)) 
                                    for tv in self.generator.trainable_variables]
            self.d_accumulators = [tf.Variable(tf.zeros_like(tv.initialized_value(), trainable=False)) 
                                    for tv in self.discriminator.trainable_variables]

            self.g_counter = tf.Variable(0.0, trainable=False)
            self.d_counter = tf.Variable(0.0, trainable=False)

    def d_loss(self, real_output, fake_output, noise_scaler=0.05):
        def random_noise(output):
            return noise_scaler * tf.random.uniform(output.shape)

        real_loss = self.cross_entropy(
            tf.ones_like(real_output) + random_noise(real_output),
            real_output)
        fake_loss = self.cross_entropy(
            tf.zeros_like(fake_output) + random_noise(fake_output),
            fake_output)

        return real_loss + fake_loss

    def g_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, audio, latent_dim, gradient_offset=2):
        noise = tf.random.normal([audio.shape[0], latent_dim])

        for _ in range(gradient_offset - 1):
            with tf.GradientTape() as gen_tape:
                generated_audio = self.generator(noise, training=True)

                real_output = self.discriminator(audio, training=False)
                fake_output = self.discriminator(generated_audio, training=False)

                gen_loss = self.g_loss(fake_output)

                self.g_metric_loss(tf.ones_like(real_output), fake_output)

            # gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            # gen_pairs = zip(gen_grads, self.generator.trainable_variables)

            # apply gradient update stochastically
            # 1 / sgd_rate chance randint returns 0
            #if random.randint(0, self.sgd_rate) == 0:
            gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            generated_audio = self.generator(noise, training=True)

            real_output = self.discriminator(audio, training=True)
            fake_output = self.discriminator(generated_audio, training=True)

            dis_loss = self.d_loss(real_output, fake_output, noise_scaler=self.d_noise)
            gen_loss = self.g_loss(fake_output)

            self.d_metric_loss(real_output, fake_output)
            self.g_metric_loss(tf.ones_like(real_output), fake_output)

        # apply gradient update stochastically
        # 1 / sgd_rate chance randint returns 0
        #if random.randint(0, self.sgd_rate) == 0:
        dis_grads = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.dis_opt.apply_gradients(zip(dis_grads, self.discriminator.trainable_variables))
        self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))


def main():
    dataset = tf.data.Dataset.from_generator(load_spectrograms, (tf.float32)).batch(BATCH_SIZE)

    gan = MusicGAN(INPUT_LENGTH, SOUND_DIM, dropout=0.20, d_lr=1e-6)

    checkpoint_dir = "./mgan-spec"
    checkpoint_freq = 200
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-spec")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gan.gen_opt,
                                     discriminator_optimizer=gan.dis_opt,
                                     generator=gan.generator,
                                     discriminator=gan.discriminator)
    if restore_latest:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for e in range(EPOCHS):
        pbar = tqdm.tqdm(total=len(filenames) // BATCH_SIZE)
        for i, batch in enumerate(dataset):
            gan.train_step(batch, LATENT_DIM, gradient_offset=1)
            description = "Epoch: {} | gen loss: {:.4f} | dis loss: {:.4f}".format(e+1, 
                                                                                   gan.g_metric_loss.result(), 
                                                                                   gan.d_metric_loss.result())
            pbar.set_description(description)

            if i % checkpoint_freq == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            pbar.update(1)

        pbar.close()

        gan.g_metric_loss.reset_states()
        gan.d_metric_loss.reset_states()


if __name__ == '__main__':
    main()
