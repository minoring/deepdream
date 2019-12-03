from absl import flags


def define_flags():
  flags.DEFINE_string('data_dir', 'data', 'Directory path of dataset')
  flags.DEFINE_string(
      'URL',
      'Green_Sea_Turtle_grazing_seagrass.jpg',
      'URL of the image')
  flags.DEFINE_integer('target_height', 240, 'Height of target image')
  flags.DEFINE_integer('target_width', 320, 'Height of target width')
  flags.DEFINE_integer('num_steps', 1000, 'Number of steps for training')
  flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
