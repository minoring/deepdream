"""Runs deepdream on the image"""
import os

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np

from flags import define_flags
from dataset import download_image
from dataset import deprocess_img
from dataset import random_roll
from model import deepdream
from model import loss_fn
from utils import save_img
from utils import create_gif

FLAGS = flags.FLAGS


def train(model, img, num_steps, learning_rate):
  """Apply gradient ascent at different scales.
  This will allow patterns generated at smaller scales to be incorporated
  into patterns at higher scales and filled in with additional detail.
  To do this you can perform the previous gradient ascent approach,
  then increase the size of the image (which is reffered to as an octave),
  and repeat this process for multiple octaves.
  """
  # Convert the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)

  for octave in range(0, FLAGS.num_octaves):
    if octave > 0:
      new_size = tf.cast(tf.convert_to_tensor(img.shape[:-1]),
                         tf.float32) * FLAGS.octave_scale
      img = tf.image.resize(img, tf.cast(new_size, tf.int32))

    for step in range(num_steps + 1):
      grads = get_tiled_gradient(model, img, FLAGS.tile_size)
      img = img + grads * learning_rate
      img = tf.clip_by_value(img, -1, 1)
      if step % 30 == 0:
        print('Octave {} Step {}'.format(octave, step))
        save_img(deprocess_img(img), step, octave)


@tf.function
def training_step(model, img, learning_rate):
  with tf.GradientTape() as tape:
    tape.watch(img)
    loss = loss_fn(img, model)
  grads = tape.gradient(loss, img)
  # Normalize the gradients.
  grads /= tf.math.reduce_std(grads) + 1e-8
  img = img + grads * learning_rate
  img = tf.clip_by_value(img, -1, 1)

  return loss, img


@tf.function
def get_tiled_gradient(model, img, tile_size=512):
  """Compute the gradient for each tile.

  When the image increases in size, so will time and memory necessary to perform
  the gradient calculation. 
  To avoid this issue, split the image into tiles and
  compute the gradient for each tile.
  Makes work on very large images, or many octaves.
  """
  shift_down, shift_right, img_rolled = random_roll(img, tile_size)
  # Initialize the image gradients to zero.
  grads = tf.zeros_like(img_rolled)

  for y in tf.range(0, img_rolled.shape[0], tile_size):
    for x in tf.range(0, img_rolled.shape[1], tile_size):
      # Calculate the gradients for this tile.
      with tf.GradientTape() as tape:
        tape.watch(img_rolled)
        # Extract a tile out of the image.
        img_tile = img_rolled[y:y + tile_size, x:x + tile_size]
        loss = loss_fn(img_tile, model)
      # Update the image gradients for this tile.
      grads = grads + tape.gradient(loss, img_rolled)
  # Undo the random shift applied to its gradients.
  grads = tf.roll(tf.roll(grads, -shift_right, axis=1), -shift_down, axis=0)
  # Normalize the gradients
  grads /= tf.math.reduce_std(grads) + 1e-8

  return grads


def main(_):
  if not os.path.isdir('sample'):
    os.mkdir('sample')
  img = download_image(FLAGS.data_dir, FLAGS.URL,
                       (FLAGS.target_height, FLAGS.target_width))
  model = deepdream()
  train(model, img, FLAGS.num_steps, FLAGS.learning_rate)
  create_gif()


if __name__ == '__main__':
  define_flags()
  app.run(main)
