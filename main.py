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


def train(model, img, num_steps, learning_rate):
  for step in range(num_steps):
    loss, img = training_step(model, img, learning_rate)

    if step % 100 == 0:
      print("Step {}, loss {}".format(step, loss))
      save_img(deprocess_img(img), step, loss)


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


def main(_):
  if not os.path.isdir('sample'):
    os.mkdir('sample')
  img = download_image(FLAGS.data_dir, FLAGS.URL,
                       (FLAGS.target_height, FLAGS.target_width))
  # Convert the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  model = deepdream()
  train(model, img, FLAGS.num_steps, FLAGS.learning_rate)


if __name__ == '__main__':
  define_flags()
  app.run(main)
