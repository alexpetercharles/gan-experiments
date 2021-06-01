import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def save_step(tensor, step):
  try:
      os.makedirs('./output/steps')
  except FileExistsError:
      pass
  image.save_img('./output/steps/' + str(step) + '.png', tf.squeeze(tensor) + 1 / 2)
