from absl import flags


def define_flags():
  flags.DEFINE_string('data_dir', 'data', 'Directory path of dataset')
  flags.DEFINE_string('URL', 'Green_Sea_Turtle_grazing_seagrass.jpg',
                      'URL of the image')
  flags.DEFINE_integer('target_height', 240, 'Height of target image')
  flags.DEFINE_integer('target_width', 320, 'Height of target width')
  flags.DEFINE_integer('num_steps', 300, 'Number of steps for training')
  flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
  flags.DEFINE_bool('scaling', True, 'Allow scaling up of the image')
  flags.DEFINE_integer('num_octaves', 3, 'Number of scaling the image')
  flags.DEFINE_float('octave_scale', 1.3, 'Increasing size of the image')
  flags.DEFINE_integer('tile_size', 512, 'Tile size of calculating gradient')
