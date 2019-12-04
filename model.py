import tensorflow as tf


def inception_v3():
  """Download pre-trained InceptionV3 image classification model"""
  return tf.keras.applications.InceptionV3(include_top=False,
                                           weights='imagenet')


def deepdream():
  base_model = inception_v3()
  # Maximize the activations of these layers
  names = ['mixed3', 'mixed5']
  layers = [base_model.get_layer(name).output for name in names]
  # Create the feature extraction model
  model = tf.keras.Model(inputs=base_model.input, outputs=layers)
  tf.keras.utils.plot_model(model, to_file='misc/model.png')
  return model


def loss_fn(img, model):
  """Calculate loss that is maximized via gradient ascent.

  The loss is normalized at each layer so the contribution from larger layers
  does not outweigh smaller layer.
  """
  # Pass forward the image through the model to retrieve the activations.
  # Converts the images into batch
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)

  return tf.reduce_sum([tf.math.reduce_mean(act) for act in layer_activations])
