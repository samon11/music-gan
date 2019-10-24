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

audio_dir = os.path.join(os.getcwd(), 'music-data')
filenames = glob.glob(audio_dir + '/fma_small/' + '/*[0-9]/*')
random.seed(8)
random.shuffle(filenames)

SR = 14400
INPUT_LENGTH = 8000
BATCH_SIZE = 1
SOUND_DIM = 8
EPOCHS = 50
LATENT_DIM = 8


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
        self.fake_output = layers.Dense(1, activation='tanh')
        
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
        out1 = self.layernorm1(attn1 + x)  # + for residual connection
        out1 = self.leaky_relu(out1)

        attn2, _ = self.mha2(out1, out1, out1)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        out2 = self.leaky_relu(out2)

        out3 = self.linear(out2)
        out3 = self.dropout3(out3, training=training)
        out3 = self.layernorm3(out3 + out2)
        out3 = self.leaky_relu(out3)

        prediction = self.flatten(out3)
        return self.out(prediction)
        

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def d_loss(real_output, fake_output, noise_scaler=0.05):
    def random_noise(output):
        return noise_scaler * tf.random.uniform(output.shape)
    
    scale = 1 / BATCH_SIZE
    real_loss = cross_entropy(tf.ones_like(real_output) + random_noise(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + random_noise(fake_output), fake_output)
    
    # (1/M) * (cross_entropy loss)
    total_loss = (1 / BATCH_SIZE) * (real_loss + fake_loss)
    return total_loss

def g_loss(fake_output):
    total_loss = (1 / BATCH_SIZE) * cross_entropy(tf.ones_like(fake_output), fake_output)
    return total_loss


d_metric_loss = tf.keras.metrics.BinaryCrossentropy(name='dis_loss')
g_metric_loss = tf.keras.metrics.BinaryCrossentropy(name='gen_loss')
@tf.function
def train_step(generator, discriminator, audio, latent_dim=8, sgd_rate=32):
    noise = tf.random.normal([audio.shape[0], latent_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated_audio = generator(noise, training=True)
        
        real_output = discriminator(audio, training=True)
        fake_output = discriminator(generated_audio, training=True)
        
        dis_loss = d_loss(real_output, fake_output)
        gen_loss = g_loss(fake_output)
        
        d_metric_loss(real_output, fake_output)
        g_metric_loss(tf.ones_like(real_output), fake_output)
    
    # apply gradient update stochastically
    # 1 / sgd_rate chance randint returns 0
    if random.randint(0, sgd_rate) == 0:
        dis_grads = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        dis_opt.apply_gradients(zip(dis_grads, discriminator.trainable_variables))
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
        

def main():
    dataset = tf.data.Dataset.from_generator(load_mp3s, (tf.float32)).batch(BATCH_SIZE)
    
    generator = Generator(INPUT_LENGTH, SOUND_DIM)
    discriminator = Discriminator(INPUT_LENGTH, SOUND_DIM)
    
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    dis_opt = tf.keras.optimizers.Adam(1e-6)

    checkpoint_dir = "./mgan"
    checkpoint_freq = 2000
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=dis_opt,
                                     generator=generator,
                                     discriminator=discriminator)
    
    for e in range(EPOCHS):
        pbar = tqdm.tqdm(total=len(filenames) // BATCH_SIZE)
        for i, batch in enumerate(dataset):
            train_step(generator, discriminator, batch, latent_dim=LATENT_DIM)
            description = "Epoch: {} | gen loss: {:.4f} | dis loss: {:.4f}".format(e+1, 
                                                                                   g_metric_loss.result(), 
                                                                                   d_metric_loss.result())
            pbar.set_description(description)

            if i % checkpoint_freq == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            pbar.update(1)

        pbar.close()

        g_metric_loss.reset_states()
        d_metric_loss.reset_states()


if __name__ == '__main__':
    main()