import tensorflow as tf
from tensorflow.keras import layers

alpha = 0.2

def define_model(input_shape):

  foundation = 4 * 4 * 256

  # sequential model means stacked layers
  model = tf.keras.models.Sequential()

  model.add(layers.Dense(foundation, input_dim = input_shape))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  model.add(layers.Reshape((4, 4, 256)))
  # (4, 4, 256)

  model.add(layers.Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same', use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (8, 8, 128)

  model.add(layers.Conv2DTranspose(64, (4, 4), strides = (4, 4), padding='same', use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (32, 32, 64)

  model.add(layers.Conv2DTranspose(32, (4, 4), strides = (4, 4), padding='same', use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (128, 128, 32)

  model.add(layers.Conv2DTranspose(16, (4, 4), strides = (2, 2), padding='same', use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha = alpha))
  # (256, 256, 16)

  model.add(layers.Conv2DTranspose(3, (4, 4), strides = (2, 2), padding='same', use_bias=False, activation='tanh'))
  # (512, 512, 3)

  return model

# print model summary
define_model(100).summary()

