import os
import numpy as np
import tensorflow as tf

from dcgan.models import discriminator, generator
from utils import image, checkpoint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

# hyper-parameters
Z_DIM = 100
D_LR = 0.0004
G_LR = 0.0004
RANDOM_SEED = 42
IMAGE_SHAPE = (512, 512, 3)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

test_z = tf.random.normal([36, Z_DIM])

def get_random_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)

# define loss function
def get_loss_fn():
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(real_logits, fake_logits):
        real_loss = criterion(tf.ones_like(real_logits), real_logits)
        fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    def g_loss_fn(fake_logits):
        return criterion(tf.ones_like(fake_logits), fake_logits)

    return d_loss_fn, g_loss_fn

# generator & discriminator
G = generator.define_model(Z_DIM)
D = discriminator.define_model(IMAGE_SHAPE)

# optimizer
g_optim = tf.keras.optimizers.Adam(G_LR, beta_1=0.5, beta_2=0.999)
d_optim = tf.keras.optimizers.Adam(D_LR, beta_1=0.5, beta_2=0.999)

# loss function
d_loss_fn, g_loss_fn = get_loss_fn()


@tf.function
def train_step(real_images, batch_size):
    z = get_random_z(Z_DIM, batch_size)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = G(z, training=True)

        fake_logits = D(fake_images, training=True)
        real_logits = D(real_images, training=True)

        d_loss = d_loss_fn(real_logits, fake_logits)
        g_loss = g_loss_fn(fake_logits)

    d_gradients = d_tape.gradient(d_loss, D.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, G.trainable_variables)

    d_optim.apply_gradients(zip(d_gradients, D.trainable_variables))
    g_optim.apply_gradients(zip(g_gradients, G.trainable_variables))

    return g_loss, d_loss

@tf.function
def fake_image_no_train():
    return G(get_random_z(Z_DIM, 1), training=False)

# training loop
def train(ds, batch_size, iteration, log_freq=20):
    ds = iter(ds)
    for step in range(iteration):
        images = next(ds)
        g_loss, d_loss = train_step(images, batch_size)

        g_loss_metrics(g_loss)
        d_loss_metrics(d_loss)
        total_loss_metrics(g_loss + d_loss)

        if step % log_freq == 0:
            template = '[{}/{}] D_loss={:.5f} G_loss={:.5f} Total_loss={:.5f}'
            print(template.format(step, iteration, d_loss_metrics.result(),
                                  g_loss_metrics.result(), total_loss_metrics.result()))
            g_loss_metrics.reset_states()
            d_loss_metrics.reset_states()
            total_loss_metrics.reset_states()
            
            image.save_step(fake_image_no_train(), step)