"""Runs deepdream on the image"""
import os
import sys

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np

from flags import define_flags
from dataset import download_image
from dataset import deprocess_img
from model import deepdream
from model import loss_fn
from utils import imshow


FLAGS = flags.FLAGS


def main(_):
  img = download_image(FLAGS.data_dir, FLAGS.URL,
                       (FLAGS.target_height, FLAGS.target_width))
  # Convert the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)

  model = deepdream()

  train(model, img, 800, 0.001)
  # result = deprocess_img(img)
  # imshow(result)
  

@tf.function
def train(model, img, num_steps, learning_rate):
  for step in range(num_steps):
    with tf.GradientTape() as tape:
      tape.watch(img)
      loss = loss_fn(img, model)
    grad = tape.gradient(loss, img)

    # Normalize the gradients.
    grad /= tf.math.reduce_std(grad) + 1e-8

    img = img + grad * learning_rate
    img = tf.clip_by_value(img, -1, 1)

    # if step % 100 == 0:
      # imshow(deprocess_img(img))
    tf.print(step, loss, output_stream=sys.stdout)


if __name__ == '__main__':
  define_flags()
  app.run(main)
