import os

import tensorflow as tf
import numpy as np


def download_image(data_dir, url, target_size=None):
  """Download an image and read it into a NumPy array."""
  file_name = url.split('/')[-1]
  file_name = os.path.join(os.path.join(os.getcwd(), data_dir, file_name))
  image_path = tf.keras.utils.get_file(file_name, origin=url)
  img = tf.keras.preprocessing.image.load_img(image_path,
                                              target_size=target_size)

  return np.array(img)


def deprocess_img(img):
  """Convert range of image from [-1, 1] into [0, 255]"""
  img = 255 * ((img + 1.0) / 2.0)
  return tf.cast(img, tf.uint8)
