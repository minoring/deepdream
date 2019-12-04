import os

import tensorflow as tf


def download_image(data_dir, url, target_size=None):
  """Download an image and read it into a NumPy array."""
  file_name = url.split('/')[-1]
  file_name = os.path.join(os.path.join(os.getcwd(), data_dir, file_name))
  image_path = tf.keras.utils.get_file(file_name, origin=url)
  img = tf.keras.preprocessing.image.load_img(image_path,
                                              target_size=target_size)

  return tf.keras.preprocessing.image.img_to_array(img)


def deprocess_img(images):
  """Convert range of image from [-1, 1] into [0, 255]"""
  images = 255 * ((images + 1.0) / 2.0)
  return tf.cast(images, tf.uint8)


def random_roll(img, maxroll):
  """Randomly roll the image to avoid tiled boundaries"""
  shift = tf.random.uniform(shape=[2],
                            minval=-maxroll,
                            maxval=maxroll,
                            dtype=tf.int32)
  shift_down, shift_right = shift[0], shift[1]
  img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)

  return shift_down, shift_right, img_rolled
