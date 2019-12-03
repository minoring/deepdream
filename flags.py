from absl import flags


def define_flags():
  flags.DEFINE_string('data_dir', 'data', 'Directory path of dataset')
  flags.DEFINE_string(
      'URL',
      'white_pekingese.jpg',
      'URL of the image')
  flags.DEFINE_integer('target_height', 320, 'Height of target image')
  flags.DEFINE_integer('target_width', 300, 'Height of target width')
  flags.DEFINE_integer('num_steps', 1000, 'Number of steps for training')
  flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
