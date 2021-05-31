import os
import tensorflow as tf

checkpoint_dir = './checktpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = ()

def define(generator_optimizer, discriminator_optimizer, generator, discriminator):
  checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

def save():
  checkpoint.save(file_prefix = checkpoint_prefix)

def restore():
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

