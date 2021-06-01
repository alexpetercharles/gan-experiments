import os
import tensorflow as tf
from pathlib import Path

# training/data
data_dir = Path('./data/')

BATCH_SIZE = 18
BUFFER_SIZE = 36
ITERATION = 100000

# data load & preprocessing
(train_x, _), (_, _) = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir, image_size=(512,512))
train_x = (train_x - 127.5) / 127.5
train_ds = (
    tf.data.Dataset.from_tensor_slices(train_x)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .repeat()
)

from wgangp.train import train

print('beginning dcgan training 💦')

train(train_ds, BATCH_SIZE, ITERATION)

print('training ended 🎉')