import tensorflow as tf


def inception_v3():
  """Download pre-trained InceptionV3 image classification model"""
  return tf.keras.applications.InceptionV3(include_top=False,
                                           weights='imagenet')


def deepdream():
  base_model = inception_v3()

  # Maximize the activations of these layers
  names = ['mixed1', 'mixed3', 'mixed5']
  layers = [base_model.get_layer(name).output for name in names]

  # Create the feature extraction model
  return tf.keras.Model(inputs=base_model.input, outputs=layers)


def loss_fn(images, model):
  """Calculate loss that is maximized via gradient ascent.

  The loss is normalized at each layer so the contribution from larger layers
  does not outweigh smaller layer.
  """
  # Pass forward the image through the model to retrieve the activations.
  # Converts the images into batch
  # img_batch = tf.stack(images, axis=0)

  layer_activations = model(images)

  loss_mix1 = tf.math.reduce_mean(layer_activations[0][0])
  loss_mix3 = tf.math.reduce_mean(layer_activations[1][1])
  loss_mix5 = tf.math.reduce_mean(layer_activations[2][2])

  
  return loss_mix1, loss_mix3, loss_mix5
