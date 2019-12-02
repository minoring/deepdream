"""Runs deepdream on the image"""
import os

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np

from flags import define_flags
from dataset import download_image
from dataset import deprocess_img
from model import deepdream
from model import loss_fn
from utils import save_img

FLAGS = flags.FLAGS


def train(model, images, num_steps, learning_rate):
  for step in range(num_steps):
    losses, images = training_step(model, images, learning_rate)

    if step % 100 == 0:
      print("Step {}, losses {}".format(step, losses))
      save_img(deprocess_img(images), step)


@tf.function
def training_step(model, images, learning_rate):
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(images)
    losses = loss_fn(images, model)
  grad_mix1 = tape.gradient(losses[0], images)[0]
  grad_mix3 = tape.gradient(losses[1], images)[1]
  grad_mix5 = tape.gradient(losses[2], images)[2]
  del tape

  # Normalize the gradients.
  # grad_mix1 = tf.math.reduce_std(grad_mix1) + 1e-8
  # grad_mix3 = tf.math.reduce_std(grad_mix3) + 1e-8
  # grad_mix5 = tf.math.reduce_std(grad_mix5) + 1e-8

  images = (images[0] + grad_mix1 * learning_rate,
            images[1] + grad_mix3 * learning_rate,
            images[2] + grad_mix5 * learning_rate)

  images = tf.clip_by_value(images, -1, 1)

  return losses, images


def main(_):
  if not os.path.isdir('sample'):
    os.mkdir('sample')

  img = download_image(FLAGS.data_dir, FLAGS.URL,
                       (FLAGS.target_height, FLAGS.target_width))
  # Convert the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)

  img_mixed1 = img.copy()
  img_mixed3 = img.copy()
  img_mixed5 = img.copy()
  images = tf.stack((img_mixed1, img_mixed3, img_mixed5), axis=0)

  model = deepdream()

  train(model, images, FLAGS.num_steps, FLAGS.learning_rate)


if __name__ == '__main__':
  define_flags()
  app.run(main)
