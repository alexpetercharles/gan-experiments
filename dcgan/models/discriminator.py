import tensorflow as tf
from tensorflow.keras import layers

alpha = 0.2
dropout = 0.3

def define_model(input_shape):

  # sequential model means stacked layers
  model = tf.keras.models.Sequential()

  model.add(layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
  model.add(layers.LeakyReLU(alpha=alpha))
  model.add(layers.Dropout(dropout))
  # (256, 256, 16)

  model.add(layers.Conv2D(32, (4, 4), strides=(4, 4), padding='same'))
  model.add(layers.LeakyReLU(alpha=alpha))
  model.add(layers.Dropout(dropout))
  # (128, 128, 32)

  model.add(layers.Conv2D(64, (4, 4), strides=(4, 4), padding='same'))
  model.add(layers.LeakyReLU(alpha=alpha))
  model.add(layers.Dropout(dropout))
  # (32, 32, 64)

  model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU(alpha=alpha))
  model.add(layers.Dropout(dropout))
  # (8, 8, 128)

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model

# print model summary
# define_model((512, 512, 3)).summary()

